void FXAI_ClearRegimeGraphQuery(FXAIRegimeGraphQuery &out)
{
   out.persistence = 0.0;
   out.transition_confidence = 0.0;
   out.instability = 0.0;
   out.edge_bias = 0.0;
   out.quality_bias = 0.0;
   out.macro_alignment = 0.0;
   out.predicted_regime = 0;
}

void FXAI_ResetRegimeGraph(void)
{
   g_regime_graph_ready = false;
   g_regime_graph_last_regime = -1;
   g_regime_graph_last_time = 0;
   for(int r0=0; r0<FXAI_REGIME_COUNT; r0++)
   {
      g_regime_graph_dwell_ema[r0] = 0.0;
      g_regime_graph_outbound_mass[r0] = 0.0;
      for(int r1=0; r1<FXAI_REGIME_COUNT; r1++)
      {
         g_regime_graph_transition_obs[r0][r1] = 0.0;
         g_regime_graph_transition_edge[r0][r1] = 0.0;
         g_regime_graph_transition_quality[r0][r1] = 0.0;
         g_regime_graph_macro_alignment[r0][r1] = 0.0;
      }
   }
}

void FXAI_RecordRegimeGraphState(const int regime_id,
                                 const datetime sample_time,
                                 const double macro_quality)
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT)
      r = 0;

   datetime t = sample_time;
   if(t <= 0)
      t = TimeCurrent();

   if(g_regime_graph_last_regime >= 0 &&
      g_regime_graph_last_regime < FXAI_REGIME_COUNT &&
      g_regime_graph_last_time > 0 &&
      t > g_regime_graph_last_time)
   {
      int prev = g_regime_graph_last_regime;
      double obs = g_regime_graph_transition_obs[prev][r];
      double alpha = FXAI_Clamp(0.14 / MathSqrt(1.0 + 0.02 * obs), 0.01, 0.14);
      double dwell_bars = (double)MathMax((int)((t - g_regime_graph_last_time) / 60), 1);
      double dwell_norm = FXAI_Clamp(dwell_bars / 30.0, 0.0, 1.0);

      g_regime_graph_transition_obs[prev][r] += 1.0;
      g_regime_graph_outbound_mass[prev] += 1.0;
      if(prev == r)
         g_regime_graph_dwell_ema[prev] =
            (1.0 - alpha) * g_regime_graph_dwell_ema[prev] + alpha * dwell_norm;
      else
         g_regime_graph_dwell_ema[prev] =
            (1.0 - alpha) * g_regime_graph_dwell_ema[prev] + alpha * 0.25 * dwell_norm;

      double macro_v = FXAI_Clamp(macro_quality, 0.0, 1.0);
      g_regime_graph_macro_alignment[prev][r] =
         (1.0 - alpha) * g_regime_graph_macro_alignment[prev][r] + alpha * macro_v;
      g_regime_graph_ready = true;
      FXAI_MarkRuntimeArtifactsDirty();
   }

   g_regime_graph_last_regime = r;
   g_regime_graph_last_time = t;
}

void FXAI_UpdateRegimeGraphFeedback(const int from_regime,
                                    const int to_regime,
                                    const double realized_edge_points,
                                    const double quality_score,
                                    const double macro_quality)
{
   int r0 = from_regime;
   if(r0 < 0 || r0 >= FXAI_REGIME_COUNT)
      return;
   int r1 = to_regime;
   if(r1 < 0 || r1 >= FXAI_REGIME_COUNT)
      r1 = r0;

   double obs = g_regime_graph_transition_obs[r0][r1];
   double alpha = FXAI_Clamp(0.16 / MathSqrt(1.0 + 0.03 * obs), 0.01, 0.16);
   double edge_norm = FXAI_Clamp(realized_edge_points / MathMax(MathAbs(realized_edge_points), 1.0), -1.0, 1.0);
   double qual = FXAI_Clamp(quality_score, 0.0, 2.0) / 2.0;
   double macro_v = FXAI_Clamp(macro_quality, 0.0, 1.0);
   g_regime_graph_transition_edge[r0][r1] =
      (1.0 - alpha) * g_regime_graph_transition_edge[r0][r1] + alpha * edge_norm;
   g_regime_graph_transition_quality[r0][r1] =
      (1.0 - alpha) * g_regime_graph_transition_quality[r0][r1] + alpha * qual;
   g_regime_graph_macro_alignment[r0][r1] =
      (1.0 - alpha) * g_regime_graph_macro_alignment[r0][r1] + alpha * macro_v;
   g_regime_graph_ready = true;
   FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_QueryRegimeGraph(const int regime_id,
                           const double macro_quality,
                           FXAIRegimeGraphQuery &out)
{
   FXAI_ClearRegimeGraphQuery(out);

   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT)
      r = 0;

   double total = g_regime_graph_outbound_mass[r];
   if(total <= 1e-6)
   {
      out.persistence = FXAI_Clamp(g_regime_graph_dwell_ema[r], 0.0, 1.0);
      out.predicted_regime = r;
      return;
   }

   double best_mass = -1.0;
   int best_regime = r;
   double edge_acc = 0.0;
   double qual_acc = 0.0;
   double macro_acc = 0.0;
   for(int nxt=0; nxt<FXAI_REGIME_COUNT; nxt++)
   {
      double obs = g_regime_graph_transition_obs[r][nxt];
      if(obs > best_mass)
      {
         best_mass = obs;
         best_regime = nxt;
      }
      double p = obs / total;
      edge_acc += p * g_regime_graph_transition_edge[r][nxt];
      qual_acc += p * g_regime_graph_transition_quality[r][nxt];
      macro_acc += p * g_regime_graph_macro_alignment[r][nxt];
   }

   out.persistence = FXAI_Clamp(g_regime_graph_transition_obs[r][r] / total, 0.0, 1.0);
   out.transition_confidence = FXAI_Clamp(best_mass / total, 0.0, 1.0);
   out.instability = FXAI_Clamp(1.0 - out.persistence, 0.0, 1.0);
   out.edge_bias = FXAI_Clamp(edge_acc, -1.0, 1.0);
   out.quality_bias = FXAI_Clamp(qual_acc, 0.0, 1.0);
   out.macro_alignment = FXAI_Clamp(0.70 * macro_acc + 0.30 * FXAI_Clamp(macro_quality, 0.0, 1.0), 0.0, 1.0);
   out.predicted_regime = best_regime;
}
