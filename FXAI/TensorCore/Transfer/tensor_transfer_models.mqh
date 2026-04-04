int FXAI_SharedTransferResolveClassLabel(const int y,
                                         const double cost_points,
                                         const double move_points)
{
   if(y >= (int)FXAI_LABEL_SELL && y <= (int)FXAI_LABEL_SKIP)
      return y;

   double cost = MathMax(cost_points, 0.0);
   double edge = MathAbs(move_points) - cost;
   double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
   if(edge <= skip_band)
      return (int)FXAI_LABEL_SKIP;
   if(y > 0)
      return (int)FXAI_LABEL_BUY;
   if(y == 0)
      return (int)FXAI_LABEL_SELL;
   return (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
}

void FXAI_ResetGlobalSharedTransferBackbone(void)
{
   g_shared_transfer_global_ready = false;
   g_shared_transfer_global_steps = 0;
   g_foundation_global_ready = false;
   g_foundation_global_steps = 0;
   g_student_global_ready = false;
   g_student_global_steps = 0;
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      g_shared_transfer_global_b[j] = 0.0;
      g_shared_transfer_global_move[j] = 0.01 * (double)(((j * 7) % 11) - 5);
      g_foundation_mask_w[j] = 0.012 * (double)(((j * 3) % 9) - 4);
      g_foundation_vol_w[j] = 0.010 * (double)(((j * 5) % 9) - 4);
      g_foundation_shift_w[j] = 0.009 * (double)(((j * 7) % 11) - 5);
      g_foundation_ctx_w[j] = 0.010 * (double)(((j * 11) % 13) - 6);
      g_student_move_w[j] = 0.008 * (double)(((j * 5) % 9) - 4);
      g_student_trade_w[j] = 0.008 * (double)(((j * 7) % 11) - 5);
      g_student_horizon_w[j] = 0.008 * (double)(((j * 9) % 13) - 6);
      g_shared_transfer_global_state_rec_w[j] = 0.18 + 0.02 * (double)((j % 5) - 2);
      g_shared_transfer_global_state_b[j] = 0.0;
      for(int c=0; c<3; c++)
      {
         g_shared_transfer_global_cls[c][j] = 0.01 * (double)((((c + 1) * (j + 5)) % 9) - 4);
         g_student_cls[c][j] = 0.008 * (double)((((c + 2) * (j + 3)) % 11) - 5);
      }
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         g_shared_transfer_global_w[j][i] = 0.0035 * (double)((((j + 2) * (i + 3)) % 13) - 6);
      for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
         g_shared_transfer_global_seq_w[j][t] = 0.0030 * (double)((((j + 5) * (t + 7)) % 17) - 8);
      for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
      {
         g_shared_transfer_global_time_w[j][c] = 0.0040 * (double)((((j + 4) * (c + 5)) % 15) - 7);
         g_shared_transfer_global_time_gate_w[j][c] = 0.0025 * (double)((((j + 6) * (c + 2)) % 13) - 6);
         g_shared_transfer_global_state_w[j][c] = 0.0035 * (double)((((j + 8) * (c + 3)) % 15) - 7);
      }
   }
   for(int d=0; d<FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS; d++)
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         g_shared_transfer_global_domain_emb[d][j] = 0.004 * (double)(((d + 3) * (j + 1)) % 7 - 3);
   for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         g_shared_transfer_global_horizon_emb[h][j] = 0.003 * (double)(((h + 5) * (j + 2)) % 9 - 4);
   for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         g_shared_transfer_global_session_emb[s][j] = 0.002 * (double)(((s + 7) * (j + 4)) % 11 - 5);
   g_foundation_mask_b = 0.0;
   g_foundation_vol_b = 0.0;
   g_foundation_shift_b = 0.0;
   g_foundation_ctx_b = 0.0;
   g_student_move_b = 0.0;
   g_student_trade_b = 0.0;
   g_student_horizon_b = 0.0;
   for(int c=0; c<3; c++)
      g_student_cls_b[c] = 0.0;
}

double FXAI_GlobalSharedTransferTrust(void)
{
   return FXAI_Clamp((double)g_shared_transfer_global_steps / 192.0, 0.0, 1.0);
}

double FXAI_GlobalFoundationTrust(void)
{
   return FXAI_Clamp((double)g_foundation_global_steps / 192.0, 0.0, 1.0);
}

double FXAI_GlobalStudentTrust(void)
{
   return FXAI_Clamp((double)g_student_global_steps / 224.0, 0.0, 1.0);
}

void FXAI_GlobalSharedTransferEncode(const double &a[],
                                     const double &x_window[][FXAI_AI_WEIGHTS],
                                     const int window_size,
                                     const double domain_hash,
                                     const int horizon_minutes,
                                     const int session_bucket,
                                     double &latent[])
{
   double seq_tokens[];
   FXAI_SharedTransferBuildSequenceTokens(a, x_window, window_size, seq_tokens);
   FXAI_SharedTransferEncodeTemporal(a,
                                     seq_tokens,
                                     x_window,
                                     window_size,
                                     FXAI_SharedTransferDomainBucket(domain_hash),
                                     FXAI_SharedTransferHorizonBucket(horizon_minutes),
                                     session_bucket,
                                     g_shared_transfer_global_w,
                                     g_shared_transfer_global_seq_w,
                                     g_shared_transfer_global_time_w,
                                     g_shared_transfer_global_time_gate_w,
                                     g_shared_transfer_global_state_w,
                                     g_shared_transfer_global_state_rec_w,
                                     g_shared_transfer_global_state_b,
                                     g_shared_transfer_global_b,
                                     g_shared_transfer_global_domain_emb,
                                     g_shared_transfer_global_horizon_emb,
                                     g_shared_transfer_global_session_emb,
                                     latent);
}

void FXAI_GlobalSharedTransferPredict(const double &a[],
                                      const double &x_window[][FXAI_AI_WEIGHTS],
                                      const int window_size,
                                      const double domain_hash,
                                      const int horizon_minutes,
                                      const int session_bucket,
                                      double &probs[],
                                      double &move_adj)
{
   double latent[];
   FXAI_GlobalSharedTransferEncode(a, x_window, window_size, domain_hash, horizon_minutes, session_bucket, latent);

   double logits[3];
   for(int c=0; c<3; c++)
   {
      logits[c] = 0.0;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         logits[c] += g_shared_transfer_global_cls[c][j] * latent[j];
   }
   FXAI_SharedTransferSoftmax(logits, probs);

   move_adj = 0.0;
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      move_adj += g_shared_transfer_global_move[j] * latent[j];
}

void FXAI_GlobalFoundationPredict(const double &a[],
                                  const double &x_window[][FXAI_AI_WEIGHTS],
                                  const int window_size,
                                  const double domain_hash,
                                  const int horizon_minutes,
                                  const int session_bucket,
                                  FXAIFoundationSignals &out)
{
   FXAI_ClearFoundationSignals(out);
   double latent[];
   FXAI_GlobalSharedTransferEncode(a,
                                   x_window,
                                   window_size,
                                   domain_hash,
                                   horizon_minutes,
                                   session_bucket,
                                   latent);

   double mask_pred = g_foundation_mask_b;
   double vol_pred = g_foundation_vol_b;
   double shift_pred = g_foundation_shift_b;
   double ctx_pred = g_foundation_ctx_b;
   double dir_bias = 0.0;
   double move_adj = 0.0;
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      mask_pred += g_foundation_mask_w[j] * latent[j];
      vol_pred += g_foundation_vol_w[j] * latent[j];
      shift_pred += g_foundation_shift_w[j] * latent[j];
      ctx_pred += g_foundation_ctx_w[j] * latent[j];
      dir_bias += latent[j] * g_shared_transfer_global_cls[(int)FXAI_LABEL_BUY][j];
      dir_bias -= latent[j] * g_shared_transfer_global_cls[(int)FXAI_LABEL_SELL][j];
      move_adj += g_shared_transfer_global_move[j] * latent[j];
   }

   out.masked_step = FXAI_ClipSym(mask_pred, 8.0);
   out.next_vol = MathExp(FXAI_ClipSym(vol_pred, 4.0));
   out.regime_transition = FXAI_Sigmoid(shift_pred);
   out.context_alignment = FXAI_Sigmoid(ctx_pred);
   out.direction_bias = FXAI_Clamp(dir_bias / (double)MathMax(FXAI_SHARED_TRANSFER_LATENT, 1), -1.0, 1.0);
   out.move_ratio = FXAI_Clamp(1.0 + 0.30 * FXAI_ClipSym(move_adj, 1.5) +
                               0.15 * FXAI_Clamp(out.next_vol / MathMax(MathAbs(out.masked_step), 0.25), 0.0, 2.0),
                               0.50,
                               2.50);
   out.tradability = FXAI_Clamp(0.18 +
                                0.20 * out.context_alignment +
                                0.18 * FXAI_Clamp(out.next_vol / 4.0, 0.0, 1.0) +
                                0.16 * (1.0 - out.regime_transition) +
                                0.12 * MathAbs(out.direction_bias),
                                0.0,
                                1.0);
   out.trust = FXAI_GlobalFoundationTrust();
   if(window_size > 0)
      out.trust = FXAI_Clamp(out.trust * (0.60 + 0.40 * FXAI_Clamp((double)window_size / 32.0, 0.0, 1.0)), 0.0, 1.0);
}

void FXAI_GlobalStudentPredict(const double &a[],
                               const double &x_window[][FXAI_AI_WEIGHTS],
                               const int window_size,
                               const double domain_hash,
                               const int horizon_minutes,
                               const int session_bucket,
                               FXAIStudentSignals &out)
{
   FXAI_ClearStudentSignals(out);
   double latent[];
   FXAI_GlobalSharedTransferEncode(a,
                                   x_window,
                                   window_size,
                                   domain_hash,
                                   horizon_minutes,
                                   session_bucket,
                                   latent);

   double logits[3];
   for(int c=0; c<3; c++)
   {
      logits[c] = g_student_cls_b[c];
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         logits[c] += g_student_cls[c][j] * latent[j];
   }
   double probs[];
   FXAI_SharedTransferSoftmax(logits, probs);
   for(int c=0; c<3; c++)
      out.class_probs[c] = (c < ArraySize(probs) ? probs[c] : (c == 2 ? 0.3334 : 0.3333));

   double move_pred = g_student_move_b;
   double trade_pred = g_student_trade_b;
   double horizon_pred = g_student_horizon_b;
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      move_pred += g_student_move_w[j] * latent[j];
      trade_pred += g_student_trade_w[j] * latent[j];
      horizon_pred += g_student_horizon_w[j] * latent[j];
   }
   out.move_ratio = FXAI_Clamp(MathExp(FXAI_ClipSym(move_pred, 4.0)), 0.25, 4.0);
   out.tradability = FXAI_Sigmoid(trade_pred);
   out.horizon_fit = FXAI_Sigmoid(horizon_pred);
   out.trust = FXAI_GlobalStudentTrust();
}

void FXAI_GlobalFoundationUpdate(const double &a[],
                                 const double &x_window[][FXAI_AI_WEIGHTS],
                                 const int window_size,
                                 const double domain_hash,
                                 const int horizon_minutes,
                                 const int session_bucket,
                                 const double masked_step_target,
                                 const double next_vol_target,
                                 const double regime_shift_target,
                                 const double context_lead_target,
                                 const double sample_w,
                                 const double lr)
{
   double latent[];
   FXAI_GlobalSharedTransferEncode(a,
                                   x_window,
                                   window_size,
                                   domain_hash,
                                   horizon_minutes,
                                   session_bucket,
                                   latent);

   double step = FXAI_Clamp(0.08 * lr * FXAI_Clamp(sample_w, 0.25, 4.0), 0.0001, 0.0080);
   double mask_pred = g_foundation_mask_b;
   double vol_pred = g_foundation_vol_b;
   double shift_pred = g_foundation_shift_b;
   double ctx_pred = g_foundation_ctx_b;
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      mask_pred += g_foundation_mask_w[j] * latent[j];
      vol_pred += g_foundation_vol_w[j] * latent[j];
      shift_pred += g_foundation_shift_w[j] * latent[j];
      ctx_pred += g_foundation_ctx_w[j] * latent[j];
   }

   double mask_err = FXAI_ClipSym(FXAI_ClipSym(masked_step_target, 8.0) - FXAI_ClipSym(mask_pred, 8.0), 4.0);
   double vol_err = FXAI_ClipSym(MathLog(MathMax(next_vol_target, 0.05)) - FXAI_ClipSym(vol_pred, 4.0), 3.0);
   double shift_out = FXAI_Sigmoid(shift_pred);
   double ctx_out = FXAI_Sigmoid(ctx_pred);
   double shift_err = FXAI_Clamp(FXAI_Clamp(regime_shift_target, 0.0, 1.0) - shift_out, -1.0, 1.0);
   double ctx_err = FXAI_Clamp(FXAI_Clamp(context_lead_target, 0.0, 1.0) - ctx_out, -1.0, 1.0);

   g_foundation_mask_b = FXAI_ClipSym(g_foundation_mask_b + 0.60 * step * mask_err, 3.0);
   g_foundation_vol_b = FXAI_ClipSym(g_foundation_vol_b + 0.50 * step * vol_err, 3.0);
   g_foundation_shift_b = FXAI_ClipSym(g_foundation_shift_b + 0.45 * step * shift_err, 3.0);
   g_foundation_ctx_b = FXAI_ClipSym(g_foundation_ctx_b + 0.45 * step * ctx_err, 3.0);
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      g_foundation_mask_w[j] = FXAI_ClipSym(g_foundation_mask_w[j] + 0.70 * step * mask_err * latent[j], 3.0);
      g_foundation_vol_w[j] = FXAI_ClipSym(g_foundation_vol_w[j] + 0.55 * step * vol_err * latent[j], 3.0);
      g_foundation_shift_w[j] = FXAI_ClipSym(g_foundation_shift_w[j] + 0.50 * step * shift_err * latent[j], 3.0);
      g_foundation_ctx_w[j] = FXAI_ClipSym(g_foundation_ctx_w[j] + 0.50 * step * ctx_err * latent[j], 3.0);
   }
   g_foundation_global_steps++;
   if(g_foundation_global_steps >= 64)
      g_foundation_global_ready = true;
}

void FXAI_GlobalStudentUpdate(const double &a[],
                              const double &x_window[][FXAI_AI_WEIGHTS],
                              const int window_size,
                              const double domain_hash,
                              const int horizon_minutes,
                              const int session_bucket,
                              const double &teacher_probs[],
                              const double teacher_move_ratio,
                              const double teacher_tradability,
                              const double teacher_horizon_fit,
                              const double sample_w,
                              const double lr)
{
   if(ArraySize(teacher_probs) < 3)
      return;

   double latent[];
   FXAI_GlobalSharedTransferEncode(a,
                                   x_window,
                                   window_size,
                                   domain_hash,
                                   horizon_minutes,
                                   session_bucket,
                                   latent);

   double logits[3];
   for(int c=0; c<3; c++)
   {
      logits[c] = g_student_cls_b[c];
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         logits[c] += g_student_cls[c][j] * latent[j];
   }
   double probs[];
   FXAI_SharedTransferSoftmax(logits, probs);

   double step = FXAI_Clamp(0.07 * lr * FXAI_Clamp(sample_w, 0.25, 4.0), 0.0001, 0.0075);
   for(int c=0; c<3; c++)
   {
      double teacher_p = (c < ArraySize(teacher_probs) ? teacher_probs[c] : 0.3333);
      double student_p = (c < ArraySize(probs) ? probs[c] : 0.3333);
      double err = FXAI_Clamp(teacher_p - student_p,
                              -1.0,
                              1.0);
      g_student_cls_b[c] = FXAI_ClipSym(g_student_cls_b[c] + 0.45 * step * err, 3.0);
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         g_student_cls[c][j] = FXAI_ClipSym(g_student_cls[c][j] + 0.60 * step * err * latent[j], 3.0);
   }

   double move_pred = g_student_move_b;
   double trade_pred = g_student_trade_b;
   double horizon_pred = g_student_horizon_b;
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      move_pred += g_student_move_w[j] * latent[j];
      trade_pred += g_student_trade_w[j] * latent[j];
      horizon_pred += g_student_horizon_w[j] * latent[j];
   }
   double move_err = FXAI_ClipSym(MathLog(MathMax(teacher_move_ratio, 0.05)) - FXAI_ClipSym(move_pred, 4.0), 3.0);
   double trade_err = FXAI_Clamp(FXAI_Clamp(teacher_tradability, 0.0, 1.0) - FXAI_Sigmoid(trade_pred), -1.0, 1.0);
   double horizon_err = FXAI_Clamp(FXAI_Clamp(teacher_horizon_fit, 0.0, 1.0) - FXAI_Sigmoid(horizon_pred), -1.0, 1.0);

   g_student_move_b = FXAI_ClipSym(g_student_move_b + 0.40 * step * move_err, 3.0);
   g_student_trade_b = FXAI_ClipSym(g_student_trade_b + 0.35 * step * trade_err, 3.0);
   g_student_horizon_b = FXAI_ClipSym(g_student_horizon_b + 0.35 * step * horizon_err, 3.0);
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      g_student_move_w[j] = FXAI_ClipSym(g_student_move_w[j] + 0.50 * step * move_err * latent[j], 3.0);
      g_student_trade_w[j] = FXAI_ClipSym(g_student_trade_w[j] + 0.45 * step * trade_err * latent[j], 3.0);
      g_student_horizon_w[j] = FXAI_ClipSym(g_student_horizon_w[j] + 0.45 * step * horizon_err * latent[j], 3.0);
   }

   g_student_global_steps++;
   if(g_student_global_steps >= 72)
      g_student_global_ready = true;
}

void FXAI_GlobalSharedTransferUpdate(const double &a[],
                                     const double &x_window[][FXAI_AI_WEIGHTS],
                                     const int window_size,
                                     const double domain_hash,
                                     const int horizon_minutes,
                                     const int session_bucket,
                                     const int y,
                                     const double cost_points,
                                     const double move_points,
                                     const double sample_w,
                                     const double lr)
{
   int cls = FXAI_SharedTransferResolveClassLabel(y, cost_points, move_points);
   double seq_tokens[];
   FXAI_SharedTransferBuildSequenceTokens(a, x_window, window_size, seq_tokens);
   double latent[];
   FXAI_GlobalSharedTransferEncode(a, x_window, window_size, domain_hash, horizon_minutes, session_bucket, latent);

   double logits[3];
   for(int c=0; c<3; c++)
   {
      logits[c] = 0.0;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         logits[c] += g_shared_transfer_global_cls[c][j] * latent[j];
   }
   double probs[];
   FXAI_SharedTransferSoftmax(logits, probs);

   double latent_grad[FXAI_SHARED_TRANSFER_LATENT];
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      latent_grad[j] = 0.0;

   double step = FXAI_Clamp(0.10 * lr * FXAI_Clamp(sample_w, 0.25, 4.0), 0.0001, 0.0100);
   for(int c=0; c<3; c++)
   {
      double target = (c == cls ? 1.0 : 0.0);
      double err = target - (ArraySize(probs) == 3 ? probs[c] : 0.3333333);
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      {
         latent_grad[j] += err * g_shared_transfer_global_cls[c][j];
         g_shared_transfer_global_cls[c][j] = FXAI_ClipSym(g_shared_transfer_global_cls[c][j] + step * err * latent[j], 3.0);
      }
   }

   double move_pred = 0.0;
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      move_pred += g_shared_transfer_global_move[j] * latent[j];
   double move_target = FXAI_Clamp(MathLog(1.0 + MathAbs(move_points)), 0.0, 4.0);
   double move_err = FXAI_ClipSym(move_target - move_pred, 3.0);
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      latent_grad[j] += 0.30 * move_err * g_shared_transfer_global_move[j];
      g_shared_transfer_global_move[j] = FXAI_ClipSym(g_shared_transfer_global_move[j] + 0.60 * step * move_err * latent[j], 3.0);
   }

   int domain_bucket = FXAI_SharedTransferDomainBucket(domain_hash);
   int horizon_bucket = FXAI_SharedTransferHorizonBucket(horizon_minutes);
   int session_idx = session_bucket;
   if(session_idx < 0) session_idx = 0;
   if(session_idx >= FXAI_PLUGIN_SESSION_BUCKETS) session_idx = FXAI_PLUGIN_SESSION_BUCKETS - 1;

   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      double g = FXAI_ClipSym(latent_grad[j] * (1.0 - latent[j] * latent[j]), 2.5);
      g_shared_transfer_global_b[j] = FXAI_ClipSym(g_shared_transfer_global_b[j] + step * g, 3.0);
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         g_shared_transfer_global_w[j][i] = FXAI_ClipSym(g_shared_transfer_global_w[j][i] + step * g * a[i], 3.0);
      for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
         g_shared_transfer_global_seq_w[j][t] = FXAI_ClipSym(g_shared_transfer_global_seq_w[j][t] + 0.80 * step * g * FXAI_SharedTransferArrayValue(seq_tokens, t, 0.0), 3.0);
      if(window_size > 0)
      {
         double pooled_den = 0.0;
         double pooled_val = FXAI_SharedTransferTemporalPoolLatent(x_window,
                                                                   window_size,
                                                                   g_shared_transfer_global_time_w,
                                                                   g_shared_transfer_global_time_gate_w,
                                                                   j);
         int bars = MathMin(window_size, FXAI_MAX_SEQUENCE_BARS);
         for(int b=0; b<bars; b++)
         {
            double bar_feats[];
            FXAI_SharedTransferExtractBarFeatures(x_window, window_size, b, bar_feats);
            double recency = 1.0 / (1.0 + 0.08 * (double)b);
            double recency_pos = 1.0 - ((double)b / (double)MathMax(bars - 1, 1));
            double gate = recency * FXAI_SharedTransferTemporalGateAt(g_shared_transfer_global_time_gate_w, j, bar_feats, recency_pos);
            pooled_den += gate;
         }
         if(pooled_den > 1e-6)
         {
            for(int b=0; b<bars; b++)
            {
               double bar_feats[];
               FXAI_SharedTransferExtractBarFeatures(x_window, window_size, b, bar_feats);
               double recency = 1.0 / (1.0 + 0.08 * (double)b);
               double recency_pos = 1.0 - ((double)b / (double)MathMax(bars - 1, 1));
               double gate = recency * FXAI_SharedTransferTemporalGateAt(g_shared_transfer_global_time_gate_w, j, bar_feats, recency_pos);
               if(gate <= 1e-6)
                  continue;
               double norm_gate = gate / pooled_den;
               double bar_val = FXAI_SharedTransferTemporalValueAt(g_shared_transfer_global_time_w, j, bar_feats);
               for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
               {
                  double feat_v = FXAI_SharedTransferArrayValue(bar_feats, c, 0.0);
                  g_shared_transfer_global_time_w[j][c] =
                     FXAI_ClipSym(g_shared_transfer_global_time_w[j][c] + 0.55 * step * g * norm_gate * feat_v, 3.0);
                  g_shared_transfer_global_time_gate_w[j][c] =
                     FXAI_ClipSym(g_shared_transfer_global_time_gate_w[j][c] +
                                  0.16 * step * g * norm_gate * (bar_val - pooled_val) * feat_v,
                                  3.0);
               }
            }
         }
      }
      double state_last = 0.0;
      double state_mean = 0.0;
      double state_abs = 0.0;
      FXAI_SharedTransferTemporalStateSummary(x_window,
                                              window_size,
                                              g_shared_transfer_global_state_w,
                                              g_shared_transfer_global_state_rec_w,
                                              g_shared_transfer_global_state_b,
                                              j,
                                              state_last,
                                              state_mean,
                                              state_abs);
      for(int c=0; c<FXAI_SHARED_TRANSFER_STATE_FEATURES; c++)
      {
         double feat_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, c);
         g_shared_transfer_global_state_w[j][c] =
            FXAI_ClipSym(g_shared_transfer_global_state_w[j][c] +
                         0.36 * step * g * FXAI_ClipSym(feat_mean, 4.0),
                         3.0);
      }
      g_shared_transfer_global_state_rec_w[j] =
         FXAI_ClipSym(g_shared_transfer_global_state_rec_w[j] +
                      0.20 * step * g * FXAI_ClipSym(state_mean + 0.35 * state_last, 2.5),
                      2.5);
      g_shared_transfer_global_state_b[j] =
         FXAI_ClipSym(g_shared_transfer_global_state_b[j] +
                      0.25 * step * g * (0.35 + 0.65 * state_abs),
                      3.0);
      g_shared_transfer_global_domain_emb[domain_bucket][j] = FXAI_ClipSym(g_shared_transfer_global_domain_emb[domain_bucket][j] + 0.45 * step * g, 3.0);
      g_shared_transfer_global_horizon_emb[horizon_bucket][j] = FXAI_ClipSym(g_shared_transfer_global_horizon_emb[horizon_bucket][j] + 0.45 * step * g, 3.0);
      g_shared_transfer_global_session_emb[session_idx][j] = FXAI_ClipSym(g_shared_transfer_global_session_emb[session_idx][j] + 0.35 * step * g, 3.0);
   }

   g_shared_transfer_global_steps++;
   if(g_shared_transfer_global_steps >= 48)
      g_shared_transfer_global_ready = true;
}
