void FXAI_ClearFoundationSignals(FXAIFoundationSignals &out)
{
   out.masked_step = 0.0;
   out.next_vol = 0.0;
   out.regime_transition = 0.0;
   out.context_alignment = 0.5;
   out.direction_bias = 0.0;
   out.move_ratio = 1.0;
   out.tradability = 0.0;
   out.trust = 0.0;
}

void FXAI_ClearStudentSignals(FXAIStudentSignals &out)
{
   out.class_probs[0] = 0.3333;
   out.class_probs[1] = 0.3333;
   out.class_probs[2] = 0.3334;
   out.move_ratio = 1.0;
   out.tradability = 0.0;
   out.horizon_fit = 0.5;
   out.trust = 0.0;
}

void FXAI_ClearAnalogMemoryQuery(FXAIAnalogMemoryQuery &out)
{
   out.similarity = 0.0;
   out.direction_agreement = 0.0;
   out.edge_norm = 0.0;
   out.quality = 0.0;
   out.path_safety = 0.5;
   out.execution_safety = 0.5;
   out.domain_alignment = 0.0;
   out.matches = 0;
}

void FXAI_ClearHierarchicalSignals(FXAIHierarchicalSignals &out)
{
   out.tradability = 0.0;
   out.direction_confidence = 0.0;
   out.move_adequacy = 0.0;
   out.path_quality = 0.0;
   out.execution_viability = 0.0;
   out.horizon_fit = 0.0;
   out.consistency = 0.0;
   out.score = 0.0;
}

double FXAI_CoreArrayValue(const double &arr[],
                           const int idx,
                           const double def_value)
{
   if(idx >= 0 && idx < ArraySize(arr))
      return arr[idx];
   return def_value;
}

int FXAI_CoreHorizonBucket(const int horizon_minutes,
                           const int bucket_count)
{
   int h = horizon_minutes;
   if(h < 1)
      h = 1;
   if(h > 720)
      h = 720;

   int buckets = bucket_count;
   if(buckets <= 1)
      return 0;

   double scaled = MathLog((double)h + 1.0) / MathLog(721.0);
   int bucket = (int)MathFloor(scaled * (double)buckets);
   if(bucket < 0)
      bucket = 0;
   if(bucket >= buckets)
      bucket = buckets - 1;
   return bucket;
}

double FXAI_AnalogFeatureByIndex(const double &x[],
                                 const int feat_idx)
{
   switch(feat_idx)
   {
      case 0:  return FXAI_Clamp(FXAI_GetInputFeature(x, 0), -4.0, 4.0);
      case 1:  return FXAI_Clamp(FXAI_GetInputFeature(x, 3), -4.0, 4.0);
      case 2:  return FXAI_Clamp(FXAI_GetInputFeature(x, 5), 0.0, 6.0);
      case 3:  return FXAI_Clamp(FXAI_GetInputFeature(x, 10), -4.0, 4.0);
      case 4:  return FXAI_Clamp(FXAI_GetInputFeature(x, 41), 0.0, 6.0);
      case 5:  return FXAI_Clamp(FXAI_GetInputFeature(x, 62), -4.0, 4.0);
      case 6:  return FXAI_Clamp(FXAI_GetInputFeature(x, 72), -4.0, 4.0);
      case 7:  return FXAI_Clamp(FXAI_GetInputFeature(x, 78), -6.0, 8.0);
      case 8:  return FXAI_Clamp(FXAI_GetInputFeature(x, 80), -4.0, 8.0);
      case 9:  return FXAI_Clamp(FXAI_GetInputFeature(x, 82), 0.0, 8.0);
      case 10: return FXAI_Clamp(0.70 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2) +
                                 0.30 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 19),
                                 0.0,
                                 1.0);
      case 11: return FXAI_Clamp(0.60 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 3) +
                                 0.40 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 15),
                                 -6.0,
                                 6.0);
      default: return 0.0;
   }
}

void FXAI_BuildAnalogMemoryVector(const double &x[],
                                  double &vec[])
{
   ArrayResize(vec, FXAI_ANALOG_MEMORY_FEATS);
   for(int i=0; i<FXAI_ANALOG_MEMORY_FEATS; i++)
      vec[i] = FXAI_AnalogFeatureByIndex(x, i);
}

double FXAI_AnalogMemoryDistanceSlot(const double &a[],
                                     const int slot)
{
   if(slot < 0 || slot >= FXAI_ANALOG_MEMORY_CAP)
      return 1e9;
   double acc = 0.0;
   for(int i=0; i<FXAI_ANALOG_MEMORY_FEATS; i++)
   {
      double wa = 1.0;
      if(i == 2 || i == 4 || i == 7 || i == 9)
         wa = 1.20;
      else if(i >= 10)
         wa = 0.85;
      double d = FXAI_CoreArrayValue(a, i, 0.0) - g_analog_memory_vec[slot][i];
      acc += wa * d * d;
   }
   return MathSqrt(MathMax(acc, 0.0));
}

void FXAI_ResetAnalogMemory(void)
{
   g_analog_memory_ready = false;
   g_analog_memory_head = 0;
   g_analog_memory_size = 0;
   for(int i=0; i<FXAI_ANALOG_MEMORY_CAP; i++)
   {
      g_analog_memory_time[i] = 0;
      g_analog_memory_regime[i] = 0;
      g_analog_memory_session[i] = 0;
      g_analog_memory_horizon[i] = 0;
      g_analog_memory_domain_hash[i] = 0.0;
      g_analog_memory_direction[i] = 0.0;
      g_analog_memory_edge_norm[i] = 0.0;
      g_analog_memory_quality[i] = 0.0;
      g_analog_memory_path_risk[i] = 0.0;
      g_analog_memory_fill_risk[i] = 0.0;
      g_analog_memory_weight[i] = 0.0;
      for(int j=0; j<FXAI_ANALOG_MEMORY_FEATS; j++)
         g_analog_memory_vec[i][j] = 0.0;
   }
}

void FXAI_UpdateAnalogMemory(const double &x[],
                             const int regime_id,
                             const int session_bucket,
                             const int horizon_minutes,
                             const double domain_hash,
                             const double move_points,
                             const double min_move_points,
                             const double quality_score,
                             const double path_risk,
                             const double fill_risk,
                             const datetime sample_time,
                             const double sample_weight)
{
   double mm = MathMax(min_move_points, 0.10);
   double direction = 0.0;
   if(MathAbs(move_points) > mm)
      direction = FXAI_Sign(move_points);

   double vec[];
   FXAI_BuildAnalogMemoryVector(x, vec);

   int slot = g_analog_memory_head;
   if(slot < 0 || slot >= FXAI_ANALOG_MEMORY_CAP)
      slot = 0;
   g_analog_memory_head = (slot + 1) % FXAI_ANALOG_MEMORY_CAP;
   if(g_analog_memory_size < FXAI_ANALOG_MEMORY_CAP)
      g_analog_memory_size++;

   g_analog_memory_time[slot] = sample_time;
   g_analog_memory_regime[slot] = (regime_id >= 0 && regime_id < FXAI_REGIME_COUNT ? regime_id : 0);
   g_analog_memory_session[slot] = (session_bucket >= 0 && session_bucket < FXAI_PLUGIN_SESSION_BUCKETS ? session_bucket : 0);
   g_analog_memory_horizon[slot] = FXAI_CoreHorizonBucket(horizon_minutes, FXAI_PLUGIN_HORIZON_BUCKETS);
   g_analog_memory_domain_hash[slot] = FXAI_Clamp(domain_hash, 0.0, 1.0);
   g_analog_memory_direction[slot] = direction;
   g_analog_memory_edge_norm[slot] = FXAI_Clamp(move_points / mm, -6.0, 6.0) / 6.0;
   g_analog_memory_quality[slot] = FXAI_Clamp(quality_score, 0.0, 2.0) / 2.0;
   g_analog_memory_path_risk[slot] = FXAI_Clamp(path_risk, 0.0, 1.0);
   g_analog_memory_fill_risk[slot] = FXAI_Clamp(fill_risk, 0.0, 1.0);
   g_analog_memory_weight[slot] = FXAI_Clamp(sample_weight, 0.10, 8.0);
   for(int j=0; j<FXAI_ANALOG_MEMORY_FEATS; j++)
      g_analog_memory_vec[slot][j] = FXAI_CoreArrayValue(vec, j, 0.0);
   g_analog_memory_ready = (g_analog_memory_size >= FXAI_ANALOG_MEMORY_MIN_MATCHES);
}

void FXAI_QueryAnalogMemory(const double &x[],
                            const int regime_id,
                            const int session_bucket,
                            const int horizon_minutes,
                            const double domain_hash,
                            FXAIAnalogMemoryQuery &out)
{
   FXAI_ClearAnalogMemoryQuery(out);
   if(g_analog_memory_size <= 0)
      return;

   double vec[];
   FXAI_BuildAnalogMemoryVector(x, vec);
   int r = (regime_id >= 0 && regime_id < FXAI_REGIME_COUNT ? regime_id : 0);
   int s = (session_bucket >= 0 && session_bucket < FXAI_PLUGIN_SESSION_BUCKETS ? session_bucket : 0);
   int h = FXAI_CoreHorizonBucket(horizon_minutes, FXAI_PLUGIN_HORIZON_BUCKETS);
   double dom = FXAI_Clamp(domain_hash, 0.0, 1.0);

   double best_sim = 0.0;
   double sum_w = 0.0;
   double sum_dir = 0.0;
   double sum_edge = 0.0;
   double sum_quality = 0.0;
   double sum_path = 0.0;
   double sum_fill = 0.0;
   double sum_dom = 0.0;
   int matches = 0;
   for(int i=0; i<g_analog_memory_size; i++)
   {
      double dist = FXAI_AnalogMemoryDistanceSlot(vec, i);
      double sim = MathExp(-0.45 * dist);
      double regime_boost = (g_analog_memory_regime[i] == r ? 1.20 : 0.85);
      double session_boost = (g_analog_memory_session[i] == s ? 1.10 : 0.92);
      double horizon_boost = (g_analog_memory_horizon[i] == h ? 1.15 : 0.90);
      double domain_align = 1.0 - MathMin(MathAbs(g_analog_memory_domain_hash[i] - dom), 1.0);
      double weight = sim *
                      regime_boost *
                      session_boost *
                      horizon_boost *
                      (0.70 + 0.30 * domain_align) *
                      (0.40 + 0.60 * FXAI_Clamp(g_analog_memory_weight[i] / 4.0, 0.0, 1.0));
      if(weight < 0.05)
         continue;

      matches++;
      if(weight > best_sim)
         best_sim = weight;
      sum_w += weight;
      sum_dir += weight * g_analog_memory_direction[i];
      sum_edge += weight * g_analog_memory_edge_norm[i];
      sum_quality += weight * g_analog_memory_quality[i];
      sum_path += weight * (1.0 - g_analog_memory_path_risk[i]);
      sum_fill += weight * (1.0 - g_analog_memory_fill_risk[i]);
      sum_dom += weight * domain_align;
   }

   if(sum_w <= 1e-9)
      return;
   out.similarity = FXAI_Clamp(best_sim, 0.0, 1.0);
   out.direction_agreement = FXAI_Clamp(sum_dir / sum_w, -1.0, 1.0);
   out.edge_norm = FXAI_Clamp(sum_edge / sum_w, -1.0, 1.0);
   out.quality = FXAI_Clamp(sum_quality / sum_w, 0.0, 1.0);
   out.path_safety = FXAI_Clamp(sum_path / sum_w, 0.0, 1.0);
   out.execution_safety = FXAI_Clamp(sum_fill / sum_w, 0.0, 1.0);
   out.domain_alignment = FXAI_Clamp(sum_dom / sum_w, 0.0, 1.0);
   out.matches = matches;
}

void FXAI_BuildHierarchicalSignals(const double &class_probs[],
                                   const double expected_move_points,
                                   const double min_move_points,
                                   const double confidence,
                                   const double reliability,
                                   const double path_risk,
                                   const double fill_risk,
                                   const double hit_time_frac,
                                   const double context_quality,
                                   const int horizon_minutes,
                                   const FXAIFoundationSignals &foundation,
                                   const FXAIStudentSignals &student,
                                   const FXAIAnalogMemoryQuery &analog,
                                   FXAIHierarchicalSignals &out)
{
   FXAI_ClearHierarchicalSignals(out);

   double mm = MathMax(min_move_points, 0.10);
   double p_sell = FXAI_Clamp(FXAI_CoreArrayValue(class_probs, (int)FXAI_LABEL_SELL, 0.3333), 0.0, 1.0);
   double p_buy = FXAI_Clamp(FXAI_CoreArrayValue(class_probs, (int)FXAI_LABEL_BUY, 0.3333), 0.0, 1.0);
   double p_skip = FXAI_Clamp(FXAI_CoreArrayValue(class_probs, (int)FXAI_LABEL_SKIP, 0.3334), 0.0, 1.0);
   double dir_gap = FXAI_Clamp(MathAbs(p_buy - p_sell), 0.0, 1.0);
   double dir_bias = FXAI_Sign(p_buy - p_sell);
   double move_ratio = FXAI_Clamp(expected_move_points / mm, 0.0, 6.0) / 6.0;
   double student_bias = FXAI_Clamp(student.class_probs[(int)FXAI_LABEL_BUY] -
                                    student.class_probs[(int)FXAI_LABEL_SELL],
                                    -1.0,
                                    1.0);
   double foundation_bias = FXAI_Clamp(foundation.direction_bias, -1.0, 1.0);
   double analog_bias = FXAI_Clamp(analog.direction_agreement, -1.0, 1.0);
   double directional_agreement = 1.0 -
                                  0.35 * MathAbs(dir_bias - FXAI_Sign(foundation_bias)) -
                                  0.30 * MathAbs(dir_bias - FXAI_Sign(student_bias)) -
                                  0.25 * MathAbs(dir_bias - FXAI_Sign(analog_bias));
   directional_agreement = FXAI_Clamp(directional_agreement, 0.0, 1.0);

   out.tradability = FXAI_Clamp(0.20 +
                                0.18 * (1.0 - p_skip) +
                                0.14 * FXAI_Clamp(confidence, 0.0, 1.0) +
                                0.12 * FXAI_Clamp(reliability, 0.0, 1.0) +
                                0.10 * FXAI_Clamp(foundation.tradability, 0.0, 1.0) +
                                0.10 * FXAI_Clamp(student.tradability, 0.0, 1.0) +
                                0.08 * FXAI_Clamp(analog.quality, 0.0, 1.0) +
                                0.06 * FXAI_Clamp(analog.similarity, 0.0, 1.0) -
                                0.10 * FXAI_Clamp(path_risk, 0.0, 1.0) -
                                0.10 * FXAI_Clamp(fill_risk, 0.0, 1.0),
                                0.0,
                                1.0);
   out.direction_confidence = FXAI_Clamp(0.35 * dir_gap +
                                         0.18 * FXAI_Clamp(confidence, 0.0, 1.0) +
                                         0.14 * FXAI_Clamp(reliability, 0.0, 1.0) +
                                         0.12 * directional_agreement +
                                         0.10 * MathAbs(foundation_bias) +
                                         0.06 * MathAbs(student_bias) +
                                         0.05 * MathAbs(analog_bias),
                                         0.0,
                                         1.0);
   out.move_adequacy = FXAI_Clamp(0.34 * move_ratio +
                                  0.20 * FXAI_Clamp(foundation.move_ratio / 2.0, 0.0, 1.0) +
                                  0.18 * FXAI_Clamp(student.move_ratio / 2.0, 0.0, 1.0) +
                                  0.16 * FXAI_Clamp(0.5 + 0.5 * analog.edge_norm, 0.0, 1.0) +
                                  0.12 * (1.0 - p_skip),
                                  0.0,
                                  1.0);
   out.path_quality = FXAI_Clamp(0.34 * (1.0 - FXAI_Clamp(path_risk, 0.0, 1.0)) +
                                 0.20 * FXAI_Clamp(analog.path_safety, 0.0, 1.0) +
                                 0.18 * FXAI_Clamp(reliability, 0.0, 1.0) +
                                 0.14 * (1.0 - FXAI_Clamp(hit_time_frac, 0.0, 1.0)) +
                                 0.14 * FXAI_Clamp(0.5 + 0.5 * context_quality, 0.0, 1.0),
                                 0.0,
                                 1.0);
   out.execution_viability = FXAI_Clamp(0.40 * (1.0 - FXAI_Clamp(fill_risk, 0.0, 1.0)) +
                                        0.24 * FXAI_Clamp(analog.execution_safety, 0.0, 1.0) +
                                        0.16 * FXAI_Clamp(foundation.tradability, 0.0, 1.0) +
                                        0.10 * FXAI_Clamp(student.tradability, 0.0, 1.0) +
                                        0.10 * FXAI_Clamp(reliability, 0.0, 1.0),
                                        0.0,
                                        1.0);
   double horizon_scale = FXAI_Clamp((double)MathMax(horizon_minutes, 1) / 240.0, 0.0, 1.0);
   out.horizon_fit = FXAI_Clamp(0.34 * FXAI_Clamp(student.horizon_fit, 0.0, 1.0) +
                                0.18 * (1.0 - MathAbs(FXAI_Clamp(hit_time_frac, 0.0, 1.0) - 0.45)) +
                                0.16 * (1.0 - 0.55 * FXAI_Clamp(foundation.regime_transition, 0.0, 1.0)) +
                                0.14 * FXAI_Clamp(analog.quality, 0.0, 1.0) +
                                0.10 * FXAI_Clamp(1.0 - horizon_scale, 0.0, 1.0) +
                                0.08 * FXAI_Clamp(foundation.context_alignment, 0.0, 1.0),
                                0.0,
                                1.0);
   out.consistency = FXAI_Clamp(0.12 +
                                0.20 * out.tradability +
                                0.18 * out.direction_confidence +
                                0.16 * out.move_adequacy +
                                0.14 * out.path_quality +
                                0.10 * out.execution_viability +
                                0.10 * out.horizon_fit +
                                0.10 * directional_agreement,
                                0.0,
                                1.0);
   out.score = FXAI_Clamp((out.tradability +
                           out.direction_confidence +
                           out.move_adequacy +
                           out.path_quality +
                           out.execution_viability +
                           out.horizon_fit +
                           out.consistency) / 7.0,
                          0.0,
                          1.0);
}

