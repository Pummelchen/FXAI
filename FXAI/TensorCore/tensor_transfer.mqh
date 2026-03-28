#ifndef __FXAI_TENSOR_TRANSFER_MQH__
#define __FXAI_TENSOR_TRANSFER_MQH__

bool   g_shared_transfer_global_ready = false;
int    g_shared_transfer_global_steps = 0;
double g_shared_transfer_global_w[FXAI_SHARED_TRANSFER_LATENT][FXAI_SHARED_TRANSFER_FEATURES];
double g_shared_transfer_global_seq_w[FXAI_SHARED_TRANSFER_LATENT][FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS];
double g_shared_transfer_global_time_w[FXAI_SHARED_TRANSFER_LATENT][FXAI_SHARED_TRANSFER_BAR_FEATURES];
double g_shared_transfer_global_time_gate_w[FXAI_SHARED_TRANSFER_LATENT][FXAI_SHARED_TRANSFER_BAR_FEATURES];
double g_shared_transfer_global_state_w[FXAI_SHARED_TRANSFER_LATENT][FXAI_SHARED_TRANSFER_STATE_FEATURES];
double g_shared_transfer_global_state_rec_w[FXAI_SHARED_TRANSFER_LATENT];
double g_shared_transfer_global_state_b[FXAI_SHARED_TRANSFER_LATENT];
double g_shared_transfer_global_b[FXAI_SHARED_TRANSFER_LATENT];
double g_shared_transfer_global_cls[3][FXAI_SHARED_TRANSFER_LATENT];
double g_shared_transfer_global_move[FXAI_SHARED_TRANSFER_LATENT];
double g_shared_transfer_global_domain_emb[FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS][FXAI_SHARED_TRANSFER_LATENT];
double g_shared_transfer_global_horizon_emb[FXAI_SHARED_TRANSFER_HORIZON_BUCKETS][FXAI_SHARED_TRANSFER_LATENT];
double g_shared_transfer_global_session_emb[FXAI_PLUGIN_SESSION_BUCKETS][FXAI_SHARED_TRANSFER_LATENT];

int FXAI_SharedTransferDomainBucket(const double domain_hash)
{
   double v = FXAI_Clamp(domain_hash, 0.0, 1.0 - 1e-9);
   int bucket = (int)MathFloor(v * (double)FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS);
   if(bucket < 0) bucket = 0;
   if(bucket >= FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS) bucket = FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS - 1;
   return bucket;
}

int FXAI_SharedTransferHorizonBucket(const int horizon_minutes)
{
   int h = horizon_minutes;
   if(h < 1) h = 1;
   if(h > 1440) h = 1440;
   int slot = 0;
   if(h <= 2) slot = 0;
   else if(h <= 5) slot = 1;
   else if(h <= 15) slot = 2;
   else if(h <= 30) slot = 3;
   else if(h <= 60) slot = 4;
   else if(h <= 240) slot = 5;
   else if(h <= 720) slot = 6;
   else slot = 7;
   if(slot < 0) slot = 0;
   if(slot >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS)
      slot = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;
   return slot;
}

double FXAI_SharedTransferWindowFeatureSegmentMean(const double &x_window[][FXAI_AI_WEIGHTS],
                                                   const int window_size,
                                                   const int feature_idx,
                                                   const int seg_start,
                                                   const int seg_len)
{
   if(window_size <= 0 || seg_len <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   int start = seg_start;
   if(start < 0) start = 0;
   if(start >= window_size) start = window_size - 1;
   int stop = start + seg_len;
   if(stop > window_size) stop = window_size;
   if(stop <= start)
      return 0.0;

   double sum = 0.0;
   int used = 0;
   for(int b=start; b<stop; b++)
   {
      sum += x_window[b][input_idx];
      used++;
   }
   if(used <= 0)
      return 0.0;
   return sum / (double)used;
}

double FXAI_SharedTransferArrayValue(const double &values[],
                                     const int index,
                                     const double fallback)
{
   int n = ArraySize(values);
   if(index < 0 || index >= n)
      return fallback;
   return values[index];
}

double FXAI_SharedTransferWindowFeatureAt(const double &x_window[][FXAI_AI_WEIGHTS],
                                          const int window_size,
                                          const int bar_idx,
                                          const int feature_idx,
                                          const double fallback)
{
   if(window_size <= 0 || bar_idx < 0 || bar_idx >= window_size)
      return fallback;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return fallback;
   return x_window[bar_idx][input_idx];
}

void FXAI_SharedTransferExtractBarFeatures(const double &x_window[][FXAI_AI_WEIGHTS],
                                           const int window_size,
                                           const int bar_idx,
                                           double &bar_feats[])
{
   ArrayResize(bar_feats, FXAI_SHARED_TRANSFER_BAR_FEATURES);
   for(int i=0; i<FXAI_SHARED_TRANSFER_BAR_FEATURES; i++)
      bar_feats[i] = 0.0;

   if(window_size <= 0 || bar_idx < 0 || bar_idx >= window_size)
      return;

   bar_feats[0] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 0, 0.0), -4.0, 4.0);
   bar_feats[1] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 3, 0.0), -4.0, 4.0);
   bar_feats[2] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 5, 0.0), 0.0, 6.0);
   bar_feats[3] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 10, 0.0), -4.0, 4.0);
   bar_feats[4] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 41, 0.0), 0.0, 6.0);
   bar_feats[5] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 62, 0.0), -4.0, 4.0);
   bar_feats[6] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 80, 0.0), -6.0, 8.0);
   bar_feats[7] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 82, 0.0), 0.0, 8.0);
   bar_feats[8] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 72, 0.0), -4.0, 4.0);
   bar_feats[9] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window, window_size, bar_idx, 78, 0.0), -4.0, 4.0);
   bar_feats[10] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window,
                                                                 window_size,
                                                                 bar_idx,
                                                                 FXAI_MACRO_EVENT_FEATURE_OFFSET + 2,
                                                                 0.0),
                              0.0,
                              1.0);
   bar_feats[11] = FXAI_Clamp(FXAI_SharedTransferWindowFeatureAt(x_window,
                                                                 window_size,
                                                                 bar_idx,
                                                                 FXAI_MACRO_EVENT_FEATURE_OFFSET + 3,
                                                                 0.0),
                              -6.0,
                              6.0);
}

double FXAI_SharedTransferTemporalGateAt(const double &gate_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                                         const int latent_idx,
                                         const double &bar_feats[],
                                         const double recency_pos)
{
   double z = 0.25 * FXAI_ClipSym(recency_pos, 1.0);
   for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
      z += gate_w[latent_idx][c] * FXAI_SharedTransferArrayValue(bar_feats, c, 0.0);
   return FXAI_Clamp(0.5 + 0.5 * FXAI_Tanh(FXAI_ClipSym(z, 6.0)), 0.0, 1.0);
}

double FXAI_SharedTransferTemporalValueAt(const double &time_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                                          const int latent_idx,
                                          const double &bar_feats[])
{
   double v = 0.0;
   for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
      v += time_w[latent_idx][c] * FXAI_SharedTransferArrayValue(bar_feats, c, 0.0);
   return FXAI_ClipSym(v, 6.0);
}

double FXAI_SharedTransferTemporalPoolLatent(const double &x_window[][FXAI_AI_WEIGHTS],
                                             const int window_size,
                                             const double &time_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                                             const double &time_gate_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                                             const int latent_idx)
{
   if(window_size <= 0 || latent_idx < 0 || latent_idx >= FXAI_SHARED_TRANSFER_LATENT)
      return 0.0;

   double pooled_num = 0.0;
   double pooled_den = 0.0;
   int bars = MathMin(window_size, FXAI_MAX_SEQUENCE_BARS);
   for(int b=0; b<bars; b++)
   {
      double bar_feats[];
      FXAI_SharedTransferExtractBarFeatures(x_window, window_size, b, bar_feats);
      double recency = 1.0 / (1.0 + 0.08 * (double)b);
      double recency_pos = 1.0 - ((double)b / (double)MathMax(bars - 1, 1));
      double gate = recency * FXAI_SharedTransferTemporalGateAt(time_gate_w, latent_idx, bar_feats, recency_pos);
      if(gate <= 1e-6)
         continue;
      pooled_num += gate * FXAI_SharedTransferTemporalValueAt(time_w, latent_idx, bar_feats);
      pooled_den += gate;
   }

   if(pooled_den <= 1e-6)
      return 0.0;
   return FXAI_ClipSym(pooled_num / pooled_den, 6.0);
}

void FXAI_SharedTransferTemporalStateSummary(const double &x_window[][FXAI_AI_WEIGHTS],
                                             const int window_size,
                                             const double &state_w[][FXAI_SHARED_TRANSFER_STATE_FEATURES],
                                             const double &state_rec_w[],
                                             const double &state_b[],
                                             const int latent_idx,
                                             double &state_last,
                                             double &state_mean,
                                             double &state_abs)
{
   state_last = 0.0;
   state_mean = 0.0;
   state_abs = 0.0;
   if(window_size <= 0 || latent_idx < 0 || latent_idx >= FXAI_SHARED_TRANSFER_LATENT)
      return;

   int bars = MathMin(window_size, FXAI_MAX_SEQUENCE_BARS);
   double carry = 0.0;
   double weighted_sum = 0.0;
   double abs_sum = 0.0;
   double weight_sum = 0.0;
   for(int rev=bars - 1; rev>=0; rev--)
   {
      double bar_feats[];
      FXAI_SharedTransferExtractBarFeatures(x_window, window_size, rev, bar_feats);
      double z = state_b[latent_idx] + state_rec_w[latent_idx] * carry;
      for(int c=0; c<FXAI_SHARED_TRANSFER_STATE_FEATURES; c++)
         z += state_w[latent_idx][c] * FXAI_SharedTransferArrayValue(bar_feats, c, 0.0);
      carry = FXAI_Tanh(FXAI_ClipSym(z, 6.0));
      double age = (double)(bars - 1 - rev);
      double recency = 1.0 / (1.0 + 0.06 * age);
      weighted_sum += recency * carry;
      abs_sum += recency * MathAbs(carry);
      weight_sum += recency;
      if(rev == 0)
         state_last = carry;
   }

   if(weight_sum <= 1e-6)
      return;
   state_mean = FXAI_ClipSym(weighted_sum / weight_sum, 6.0);
   state_abs = FXAI_Clamp(abs_sum / weight_sum, 0.0, 1.0);
}

void FXAI_SharedTransferBuildSequenceTokens(const double &a[],
                                           const double &x_window[][FXAI_AI_WEIGHTS],
                                           const int window_size,
                                           double &tokens[])
{
   ArrayResize(tokens, FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS);
   for(int i=0; i<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; i++)
      tokens[i] = 0.0;

   if(window_size <= 0)
   {
      tokens[0] = FXAI_SharedTransferArrayValue(a, 20, 0.0);
      tokens[1] = FXAI_SharedTransferArrayValue(a, 21, 0.0);
      tokens[2] = FXAI_SharedTransferArrayValue(a, 22, 0.0);
      tokens[3] = FXAI_SharedTransferArrayValue(a, 23, 0.0);
      tokens[4] = FXAI_SharedTransferArrayValue(a, 24, 0.0);
      tokens[5] = FXAI_SharedTransferArrayValue(a, 25, 0.0);
      tokens[6] = FXAI_SharedTransferArrayValue(a, 26, 0.0);
      tokens[7] = FXAI_SharedTransferArrayValue(a, 27, 0.0);
      tokens[8] = FXAI_Clamp(0.50 * FXAI_SharedTransferArrayValue(a, 11, 0.0) + 0.50 * FXAI_SharedTransferArrayValue(a, 12, 0.0), -4.0, 4.0);
      tokens[9] = FXAI_Clamp(0.55 * FXAI_SharedTransferArrayValue(a, 13, 0.0) + 0.45 * FXAI_SharedTransferArrayValue(a, 14, 0.0), -4.0, 4.0);
      tokens[10] = FXAI_Clamp(0.50 * FXAI_SharedTransferArrayValue(a, 15, 0.0) + 0.50 * FXAI_SharedTransferArrayValue(a, 16, 0.0), -4.0, 4.0);
      tokens[11] = FXAI_Clamp(0.50 * FXAI_SharedTransferArrayValue(a, 17, 0.0) + 0.50 * FXAI_SharedTransferArrayValue(a, 18, 0.0), -4.0, 4.0);
      tokens[12] = FXAI_Clamp(0.60 * FXAI_SharedTransferArrayValue(a, 6, 0.0) + 0.40 * FXAI_SharedTransferArrayValue(a, 7, 0.0), -4.0, 4.0);
      tokens[13] = FXAI_Clamp(0.50 * FXAI_SharedTransferArrayValue(a, 4, 0.0) + 0.50 * FXAI_SharedTransferArrayValue(a, 8, 0.0), -4.0, 4.0);
      tokens[14] = FXAI_Clamp(FXAI_SharedTransferArrayValue(a, 19, 0.0), -4.0, 4.0);
      tokens[15] = FXAI_Clamp((MathAbs(tokens[0]) + MathAbs(tokens[3]) + MathAbs(tokens[7])) / 3.0, 0.0, 6.0);
      return;
   }

   int seg = MathMax(window_size / 4, 2);
   if(seg > window_size) seg = window_size;
   int tail_start = MathMax(window_size - seg, 0);
   double ret_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 0, 0, seg);
   double ret_mid = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 1, seg, seg);
   double ret_tail = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 2, tail_start, seg);
   double trend_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 3, 0, seg);
   double vol_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 5, 0, seg);
   double vol_tail = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 5, tail_start, seg);
   double spread_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 80, 0, seg);
   double spread_tail = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 80, tail_start, seg);
   double spread_z_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 81, 0, seg);
   double ctx_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 10, 0, seg);
   double ctx_tail = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 10, tail_start, seg);
   double shared_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 62, 0, seg);
   double shared_lead_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 64, 0, seg);
   double shared_cover_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 65, 0, seg);
   double session_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 72, 0, seg);
   double overlap_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 73, 0, seg);
   double rollover_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 74, 0, seg);
   double carry_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 78, 0, seg);
   double drift_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 79, 0, seg);
   double drift_tail = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, 79, tail_start, seg);
   double macro_pre_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0, 0, seg);
   double macro_post_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1, 0, seg);
   double macro_imp_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2, 0, seg);
   double macro_surprise_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 3, 0, seg);
   double macro_abs_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4, 0, seg);
   double macro_z_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 6, 0, seg);
   double macro_revision_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 7, 0, seg);
   double macro_currency_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 8, 0, seg);
   double macro_provenance_recent = FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 9, 0, seg);

   double main_mtf_body_recent = 0.0;
   double main_mtf_spread_recent = 0.0;
   double ctx_mtf_body_recent = 0.0;
   double ctx_mtf_spread_recent = 0.0;
   for(int tf_slot=0; tf_slot<FXAI_MAIN_MTF_TF_COUNT; tf_slot++)
   {
      int base = FXAI_MainMTFFeatureIndex(tf_slot, 0);
      if(base < 0)
         continue;
      main_mtf_body_recent += FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, base + 0, 0, seg);
      main_mtf_spread_recent += FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, base + 3, 0, seg);
   }
   main_mtf_body_recent /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   main_mtf_spread_recent /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   int ctx_used = 0;
   for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
   {
      for(int tf_slot=0; tf_slot<FXAI_CONTEXT_MTF_TF_COUNT; tf_slot++)
      {
         int base = FXAI_ContextMTFFeatureIndex(slot, tf_slot, 0);
         if(base < 0)
            continue;
         ctx_mtf_body_recent += FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, base + 0, 0, seg);
         ctx_mtf_spread_recent += FXAI_SharedTransferWindowFeatureSegmentMean(x_window, window_size, base + 3, 0, seg);
         ctx_used++;
      }
   }
   if(ctx_used > 0)
   {
      ctx_mtf_body_recent /= (double)ctx_used;
      ctx_mtf_spread_recent /= (double)ctx_used;
   }

   tokens[0] = FXAI_Clamp(ret_recent, -4.0, 4.0);
   tokens[1] = FXAI_Clamp(ret_mid, -4.0, 4.0);
   tokens[2] = FXAI_Clamp(ret_tail, -4.0, 4.0);
   tokens[3] = FXAI_Clamp(trend_recent + 0.35 * (ret_recent - ret_tail), -4.0, 4.0);
   tokens[4] = FXAI_Clamp(0.65 * vol_recent + 0.35 * (vol_recent - vol_tail), 0.0, 6.0);
   tokens[5] = FXAI_Clamp(0.60 * spread_recent + 0.25 * spread_z_recent + 0.15 * (spread_recent - spread_tail), -6.0, 8.0);
   tokens[6] = FXAI_Clamp(0.60 * ctx_recent + 0.20 * (ctx_recent - ctx_tail) + 0.20 * shared_recent, -4.0, 4.0);
   tokens[7] = FXAI_Clamp(0.45 * shared_recent + 0.30 * shared_lead_recent + 0.25 * shared_cover_recent, -4.0, 4.0);
   tokens[8] = FXAI_Clamp(0.45 * session_recent + 0.30 * overlap_recent + 0.25 * rollover_recent, -4.0, 4.0);
   tokens[9] = FXAI_Clamp(0.45 * carry_recent + 0.35 * drift_recent + 0.20 * (drift_recent - drift_tail), -4.0, 4.0);
   tokens[10] = FXAI_Clamp(0.25 * macro_pre_recent +
                           0.12 * macro_post_recent +
                           0.22 * macro_imp_recent +
                           0.18 * macro_surprise_recent +
                           0.13 * macro_z_recent +
                           0.10 * macro_currency_recent,
                           -6.0,
                           6.0);
   tokens[11] = FXAI_Clamp(0.28 * macro_abs_recent +
                           0.15 * macro_revision_recent +
                           0.10 * macro_provenance_recent +
                           0.27 * main_mtf_body_recent +
                           0.20 * ctx_mtf_body_recent,
                           -4.0,
                           6.0);
   tokens[12] = FXAI_Clamp(main_mtf_spread_recent - ctx_mtf_spread_recent, -4.0, 4.0);
   tokens[13] = FXAI_Clamp(ret_recent - ret_tail, -4.0, 4.0);
   tokens[14] = FXAI_Clamp(spread_recent - spread_tail, -6.0, 6.0);
   tokens[15] = FXAI_Clamp(0.40 * FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 5) +
                           0.35 * FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 80) +
                           0.25 * FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 79),
                           0.0,
                           6.0);
}

void FXAI_SharedTransferEncodeTemporal(const double &a[],
                                       const double &seq_tokens[],
                                       const double &x_window[][FXAI_AI_WEIGHTS],
                                       const int window_size,
                                       const int domain_bucket,
                                       const int horizon_bucket,
                                       const int session_bucket,
                                       const double &w[][FXAI_SHARED_TRANSFER_FEATURES],
                                       const double &seq_w[][FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS],
                                       const double &time_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                                       const double &time_gate_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                                       const double &state_w[][FXAI_SHARED_TRANSFER_STATE_FEATURES],
                                       const double &state_rec_w[],
                                       const double &state_b[],
                                       const double &b[],
                                       const double &domain_emb[][FXAI_SHARED_TRANSFER_LATENT],
                                       const double &horizon_emb[][FXAI_SHARED_TRANSFER_LATENT],
                                       const double &session_emb[][FXAI_SHARED_TRANSFER_LATENT],
                                       double &latent[])
{
   ArrayResize(latent, FXAI_SHARED_TRANSFER_LATENT);
   int db = domain_bucket;
   if(db < 0) db = 0;
   if(db >= FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS) db = FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS - 1;
   int hb = horizon_bucket;
   if(hb < 0) hb = 0;
   if(hb >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS) hb = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;
   int sb = session_bucket;
   if(sb < 0) sb = 0;
   if(sb >= FXAI_PLUGIN_SESSION_BUCKETS) sb = FXAI_PLUGIN_SESSION_BUCKETS - 1;

   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      double state_last = 0.0;
      double state_mean = 0.0;
      double state_abs = 0.0;
      FXAI_SharedTransferTemporalStateSummary(x_window,
                                              window_size,
                                              state_w,
                                              state_rec_w,
                                              state_b,
                                              j,
                                              state_last,
                                              state_mean,
                                              state_abs);
      double z = b[j] +
                 domain_emb[db][j] +
                 horizon_emb[hb][j] +
                 session_emb[sb][j];
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         z += w[j][i] * a[i];
      for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
         z += seq_w[j][t] * FXAI_SharedTransferArrayValue(seq_tokens, t, 0.0);
      z += FXAI_SharedTransferTemporalPoolLatent(x_window, window_size, time_w, time_gate_w, j);
      z += 0.65 * state_last + 0.25 * state_mean + 0.18 * (2.0 * state_abs - 1.0);
      latent[j] = FXAI_Tanh(FXAI_ClipSym(z, 6.0));
   }
}

void FXAI_SharedTransferEncode(const double &a[],
                               const double &x_window[][FXAI_AI_WEIGHTS],
                               const int window_size,
                               const int domain_bucket,
                               const int horizon_bucket,
                               const int session_bucket,
                               const double &w[][FXAI_SHARED_TRANSFER_FEATURES],
                               const double &time_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                               const double &time_gate_w[][FXAI_SHARED_TRANSFER_BAR_FEATURES],
                               const double &state_w[][FXAI_SHARED_TRANSFER_STATE_FEATURES],
                               const double &state_rec_w[],
                               const double &state_b[],
                               const double &b[],
                               const double &domain_emb[][FXAI_SHARED_TRANSFER_LATENT],
                               const double &horizon_emb[][FXAI_SHARED_TRANSFER_LATENT],
                               const double &session_emb[][FXAI_SHARED_TRANSFER_LATENT],
                               double &latent[])
{
   double seq_tokens[];
   ArrayResize(seq_tokens, FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS);
   for(int i=0; i<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; i++)
      seq_tokens[i] = 0.0;
   double seq_w_zero[FXAI_SHARED_TRANSFER_LATENT][FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS];
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
         seq_w_zero[j][t] = 0.0;
   FXAI_SharedTransferEncodeTemporal(a,
                                     seq_tokens,
                                     x_window,
                                     window_size,
                                     domain_bucket,
                                     horizon_bucket,
                                     session_bucket,
                                     w,
                                     seq_w_zero,
                                     time_w,
                                     time_gate_w,
                                     state_w,
                                     state_rec_w,
                                     state_b,
                                     b,
                                     domain_emb,
                                     horizon_emb,
                                     session_emb,
                                     latent);
}

void FXAI_SharedTransferSoftmax(const double &logits[],
                                double &probs[])
{
   ArrayResize(probs, 3);
   double mx = logits[0];
   if(logits[1] > mx) mx = logits[1];
   if(logits[2] > mx) mx = logits[2];

   double den = 0.0;
   for(int c=0; c<3; c++)
   {
      probs[c] = MathExp(FXAI_ClipSym(logits[c] - mx, 10.0));
      den += probs[c];
   }
   if(den <= 0.0) den = 1.0;
   for(int c=0; c<3; c++)
      probs[c] /= den;
}

double FXAI_SharedTransferWindowFeatureMean(const double &x_window[][FXAI_AI_WEIGHTS],
                                            const int window_size,
                                            const int feature_idx)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   double sum = 0.0;
   int used = 0;
   for(int b=0; b<window_size; b++)
   {
      sum += x_window[b][input_idx];
      used++;
   }
   if(used <= 0)
      return 0.0;
   return sum / (double)used;
}

double FXAI_SharedTransferWindowFeatureEMAMean(const double &x_window[][FXAI_AI_WEIGHTS],
                                               const int window_size,
                                               const int feature_idx,
                                               const double decay = 0.72)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   double a = FXAI_Clamp(decay, 0.05, 0.98);
   double w = 1.0;
   double sw = 0.0;
   double sum = 0.0;
   for(int b=0; b<window_size; b++)
   {
      sum += w * x_window[b][input_idx];
      sw += w;
      w *= a;
   }
   if(sw <= 0.0)
      return 0.0;
   return sum / sw;
}

double FXAI_SharedTransferWindowFeatureStd(const double &x_window[][FXAI_AI_WEIGHTS],
                                           const int window_size,
                                           const int feature_idx)
{
   if(window_size <= 1)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   double mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, feature_idx);
   double acc = 0.0;
   for(int b=0; b<window_size; b++)
   {
      double d = x_window[b][input_idx] - mean;
      acc += d * d;
   }
   return MathSqrt(acc / (double)MathMax(window_size, 1));
}

double FXAI_SharedTransferWindowFeatureRange(const double &x_window[][FXAI_AI_WEIGHTS],
                                             const int window_size,
                                             const int feature_idx,
                                             const int recent_bars = 0)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   int n = recent_bars;
   if(n <= 0 || n > window_size)
      n = window_size;
   double lo = x_window[0][input_idx];
   double hi = lo;
   for(int b=0; b<n; b++)
   {
      double v = x_window[b][input_idx];
      if(v < lo)
         lo = v;
      if(v > hi)
         hi = v;
   }
   return hi - lo;
}

double FXAI_SharedTransferWindowFeatureSlope(const double &x_window[][FXAI_AI_WEIGHTS],
                                             const int window_size,
                                             const int feature_idx)
{
   if(window_size <= 1)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;
   double first = x_window[0][input_idx];
   double last = x_window[window_size - 1][input_idx];
   return (first - last) / (double)MathMax(window_size - 1, 1);
}

double FXAI_SharedTransferWindowFeatureRecentDelta(const double &x_window[][FXAI_AI_WEIGHTS],
                                                   const int window_size,
                                                   const int feature_idx,
                                                   const int recent_bars)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   int n = recent_bars;
   if(n <= 1)
      n = MathMax(window_size / 4, 2);
   if(n > window_size)
      n = window_size;
   int last_idx = n - 1;
   if(last_idx < 0)
      last_idx = 0;
   return x_window[0][input_idx] - x_window[last_idx][input_idx];
}

void FXAI_BuildSharedTransferInputGlobalBase(const double &x[],
                                             const double domain_hash,
                                             const int horizon_minutes,
                                             double &out[])
{
   ArrayResize(out, FXAI_SHARED_TRANSFER_FEATURES);
   for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
      out[i] = 0.0;
   out[0] = 1.0;
   out[1] = FXAI_GetInputFeature(x, 62);
   out[2] = FXAI_GetInputFeature(x, 63);
   out[3] = FXAI_GetInputFeature(x, 64);
   out[4] = FXAI_Clamp(0.5 + 0.5 * FXAI_GetInputFeature(x, 65), 0.0, 1.0);

   double ret_mix = 0.0;
   double lag_mix = 0.0;
   double rel_mix = 0.0;
   double corr_mix = 0.0;
   double weight_total = 0.0;
   for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
   {
      int base = 50 + slot * 4;
      double ctx_ret = FXAI_GetInputFeature(x, base + 0);
      double ctx_lag = FXAI_GetInputFeature(x, base + 1);
      double ctx_rel = FXAI_GetInputFeature(x, base + 2);
      double ctx_corr = FXAI_GetInputFeature(x, base + 3);
      double w = FXAI_Clamp((0.30 + 0.70 * MathAbs(ctx_corr)) *
                            (0.35 + 0.65 * out[4]) *
                            (0.35 + 0.25 * MathAbs(ctx_ret) + 0.25 * MathAbs(ctx_lag) + 0.15 * MathAbs(ctx_rel)),
                            0.0,
                            3.0);
      if(w <= 1e-6)
         continue;
      ret_mix += w * ctx_ret;
      lag_mix += w * ctx_lag;
      rel_mix += w * ctx_rel;
      corr_mix += w * ctx_corr;
      weight_total += w;
   }
   if(weight_total > 1e-6)
   {
      out[5] = ret_mix / weight_total;
      out[6] = lag_mix / weight_total;
      out[7] = rel_mix / weight_total;
      out[8] = corr_mix / weight_total;
   }
   else
   {
      out[5] = 0.0;
      out[6] = 0.0;
      out[7] = 0.0;
      out[8] = 0.0;
   }

   double domain = FXAI_Clamp(domain_hash, 0.0, 1.0);
   double horizon_scale = FXAI_Clamp(MathLog(1.0 + (double)MathMax(horizon_minutes, 1)) / MathLog(1.0 + 1440.0), 0.0, 1.0);
   double main_mtf_body = 0.0;
   double main_mtf_loc = 0.0;
   double main_mtf_range = 0.0;
   double main_mtf_spread = 0.0;
   for(int tf_slot=0; tf_slot<FXAI_MAIN_MTF_TF_COUNT; tf_slot++)
   {
      int base = FXAI_MainMTFFeatureIndex(tf_slot, 0);
      if(base < 0)
         continue;
      main_mtf_body += FXAI_GetInputFeature(x, base + 0);
      main_mtf_loc += FXAI_GetInputFeature(x, base + 1);
      main_mtf_range += FXAI_GetInputFeature(x, base + 2);
      main_mtf_spread += FXAI_GetInputFeature(x, base + 3);
   }
   main_mtf_body /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   main_mtf_loc /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   main_mtf_range /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   main_mtf_spread /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);

   double ctx_mtf_body = 0.0;
   double ctx_mtf_loc = 0.0;
   double ctx_mtf_range = 0.0;
   double ctx_mtf_spread = 0.0;
   double ctx_mtf_weight = 0.0;
   for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
   {
      double slot_corr = MathAbs(FXAI_GetInputFeature(x, 50 + slot * 4 + 3));
      double slot_weight = 0.35 + 0.65 * slot_corr;
      double slot_body = 0.0;
      double slot_loc = 0.0;
      double slot_range = 0.0;
      double slot_spread = 0.0;
      int slot_used = 0;
      for(int tf_slot=0; tf_slot<FXAI_CONTEXT_MTF_TF_COUNT; tf_slot++)
      {
         int base = FXAI_ContextMTFFeatureIndex(slot, tf_slot, 0);
         if(base < 0)
            continue;
         slot_body += FXAI_GetInputFeature(x, base + 0);
         slot_loc += FXAI_GetInputFeature(x, base + 1);
         slot_range += FXAI_GetInputFeature(x, base + 2);
         slot_spread += FXAI_GetInputFeature(x, base + 3);
         slot_used++;
      }
      if(slot_used <= 0)
         continue;
      slot_body /= (double)slot_used;
      slot_loc /= (double)slot_used;
      slot_range /= (double)slot_used;
      slot_spread /= (double)slot_used;
      ctx_mtf_body += slot_weight * slot_body;
      ctx_mtf_loc += slot_weight * slot_loc;
      ctx_mtf_range += slot_weight * slot_range;
      ctx_mtf_spread += slot_weight * slot_spread;
      ctx_mtf_weight += slot_weight;
   }
   if(ctx_mtf_weight > 1e-6)
   {
      ctx_mtf_body /= ctx_mtf_weight;
      ctx_mtf_loc /= ctx_mtf_weight;
      ctx_mtf_range /= ctx_mtf_weight;
      ctx_mtf_spread /= ctx_mtf_weight;
   }

   double macro_pre = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0);
   double macro_post = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1);
   double macro_imp = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2);
   double macro_surprise = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 3);
   double macro_surprise_abs = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4);
   double macro_class = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 5);
   double macro_surprise_z = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 6);
   double macro_revision_abs = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 7);
   double macro_currency_rel = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 8);
   double macro_provenance = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 9);

   out[9] = 2.0 * domain - 1.0;
   out[10] = 2.0 * horizon_scale - 1.0;
   out[11] = FXAI_Clamp(0.60 * FXAI_GetInputFeature(x, 72) +
                        0.15 * FXAI_GetInputFeature(x, 73) +
                        0.10 * macro_pre -
                        0.08 * macro_post +
                        0.07 * main_mtf_loc +
                        0.06 * ctx_mtf_loc,
                        -1.0,
                        1.0);
   out[12] = FXAI_Clamp(0.28 * FXAI_GetInputFeature(x, 74) +
                        0.16 * FXAI_GetInputFeature(x, 75) +
                        0.14 * FXAI_GetInputFeature(x, 78) +
                        0.10 * macro_imp +
                        0.08 * macro_class +
                        0.08 * main_mtf_body +
                        0.06 * ctx_mtf_body,
                        -1.0,
                        1.0);
   out[13] = FXAI_Clamp(0.16 * FXAI_GetInputFeature(x, 76) -
                        0.16 * FXAI_GetInputFeature(x, 77) -
                        0.10 * FXAI_GetInputFeature(x, 79) +
                        0.08 * FXAI_GetInputFeature(x, 6) +
                        0.08 * FXAI_GetInputFeature(x, 81) +
                        0.06 * macro_surprise_abs -
                        0.05 * main_mtf_spread -
                        0.05 * FXAI_GetInputFeature(x, 82) +
                        0.04 * ctx_mtf_spread,
                        -4.0,
                        4.0);
   out[14] = FXAI_Clamp(0.42 * FXAI_GetInputFeature(x, 18) +
                        0.18 * FXAI_GetInputFeature(x, 19) -
                        0.18 * FXAI_GetInputFeature(x, 20) +
                        0.12 * FXAI_GetInputFeature(x, 21) +
                        0.06 * macro_imp +
                        0.08 * main_mtf_body +
                        0.08 * main_mtf_loc,
                        -4.0,
                        4.0);
   out[15] = FXAI_Clamp(0.16 * FXAI_GetInputFeature(x, 66) +
                        0.12 * FXAI_GetInputFeature(x, 67) +
                        0.12 * FXAI_GetInputFeature(x, 68) +
                        0.10 * FXAI_GetInputFeature(x, 69) +
                        0.14 * FXAI_GetInputFeature(x, 71) +
                        0.10 * macro_surprise +
                        0.08 * macro_surprise_abs +
                        0.06 * FXAI_GetInputFeature(x, 81) +
                        0.04 * FXAI_GetInputFeature(x, 83) +
                        0.05 * main_mtf_range +
                        0.05 * ctx_mtf_range,
                        -6.0,
                        6.0);
   out[16] = FXAI_Clamp(0.48 * FXAI_GetInputFeature(x, 68) +
                        0.18 * FXAI_GetInputFeature(x, 81) +
                        0.12 * FXAI_GetInputFeature(x, 80) +
                        0.10 * macro_imp +
                        0.08 * macro_surprise_abs +
                        0.08 * main_mtf_spread,
                        -4.0,
                        8.0);
   out[17] = FXAI_Clamp(0.55 * FXAI_GetInputFeature(x, 70) +
                        0.18 * FXAI_GetInputFeature(x, 82) +
                        0.10 * macro_post +
                        0.08 * macro_surprise_abs +
                        0.05 * main_mtf_range +
                        0.04 * MathAbs(FXAI_GetInputFeature(x, 83)),
                        0.0,
                        8.0);
   out[18] = FXAI_Clamp(0.45 * macro_surprise +
                        0.20 * macro_class +
                        0.15 * FXAI_GetInputFeature(x, 78) +
                        0.10 * FXAI_GetInputFeature(x, 72) +
                        0.10 * FXAI_GetInputFeature(x, 73) +
                        0.12 * macro_surprise_z +
                        0.08 * macro_currency_rel,
                        -6.0,
                        6.0);
   out[19] = FXAI_Clamp(0.40 * macro_surprise_abs +
                        0.22 * macro_imp +
                        0.18 * macro_pre +
                        0.10 * macro_post +
                        0.10 * FXAI_GetInputFeature(x, 79) +
                        0.10 * macro_revision_abs +
                        0.08 * macro_provenance,
                        0.0,
                        6.0);
}

void FXAI_BuildSharedTransferInputGlobal(const double &x[],
                                         const double domain_hash,
                                         const int horizon_minutes,
                                         double &out[])
{
   FXAI_BuildSharedTransferInputGlobalBase(x, domain_hash, horizon_minutes, out);
}

void FXAI_BuildSharedTransferInputGlobal(const double &x[],
                                         const double &x_window[][FXAI_AI_WEIGHTS],
                                         const int window_size,
                                         const double domain_hash,
                                         const int horizon_minutes,
                                         double &out[])
{
   FXAI_BuildSharedTransferInputGlobalBase(x, domain_hash, horizon_minutes, out);
   if(ArraySize(out) < FXAI_SHARED_TRANSFER_FEATURES)
      ArrayResize(out, FXAI_SHARED_TRANSFER_FEATURES);

   if(window_size <= 0)
   {
      out[20] = FXAI_GetInputFeature(x, 0);
      out[21] = FXAI_GetInputFeature(x, 3);
      out[22] = FXAI_GetInputFeature(x, 41);
      out[23] = 0.50 * FXAI_GetInputFeature(x, 68) + 0.25 * FXAI_GetInputFeature(x, 69) + 0.25 * FXAI_GetInputFeature(x, 80);
      out[24] = 0.35 * FXAI_GetInputFeature(x, 10) + 0.25 * FXAI_GetInputFeature(x, 62) + 0.20 * FXAI_GetInputFeature(x, 63) + 0.20 * FXAI_GetInputFeature(x, 64);
      out[25] = 0.25 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0) +
                0.15 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1) +
                0.20 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2) +
                0.20 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4) +
                0.20 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 5);
      out[26] = 0.20 * FXAI_GetInputFeature(x, 72) +
                0.15 * FXAI_GetInputFeature(x, 73) +
                0.20 * FXAI_GetInputFeature(x, 74) +
                0.10 * FXAI_GetInputFeature(x, 75) +
                0.15 * FXAI_GetInputFeature(x, 78) +
                0.10 * (FXAI_GetInputFeature(x, 76) - FXAI_GetInputFeature(x, 77)) +
                0.10 * FXAI_GetInputFeature(x, 79);
      out[27] = 0.30 * FXAI_GetInputFeature(x, 79) +
                0.25 * MathAbs(FXAI_GetInputFeature(x, 63)) +
                0.20 * MathAbs(FXAI_GetInputFeature(x, 68)) +
                0.15 * FXAI_GetInputFeature(x, 82) +
                0.10 * MathAbs(FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4));
      return;
   }

   double ret_fast = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 0);
   double ret_mid = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 1);
   double ret_long = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 2);
   double slope_m1 = FXAI_SharedTransferWindowFeatureSlope(x_window, window_size, 3);
   double vol_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 5);
   double vol_std = FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 5);
   double atr_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 41);
   double ctx_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 10);
   double ctx_std = FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 10);
   double shared_util = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 62);
   double shared_stability = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 63);
   double shared_lead = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 64);
   double shared_coverage = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 65);
   double spread_shock = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 68);
   double spread_accel = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 69);
   double spread_to_range = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 70);
   double micro_trend = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 71);
   double session_transition = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 72);
   double session_overlap = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 73);
   double rollover_prox = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 74);
   double triple_swap = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 75);
   double swap_bias = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 76) -
                      FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 77);
   double carry_align = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 78);
   double drift_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 79);
   double spread_log = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 80);
   double spread_z = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 81);
   double spread_vol = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 82);
   double spread_rank = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 83);
   double macro_pre = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0);
   double macro_post = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1);
   double macro_imp = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2);
   double macro_surprise = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 3);
   double macro_surprise_abs = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4);
   double macro_class = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 5);
   double ret_delta = FXAI_SharedTransferWindowFeatureRecentDelta(x_window, window_size, 0, MathMax(window_size / 4, 3));
   double spread_delta = FXAI_SharedTransferWindowFeatureRecentDelta(x_window, window_size, 80, MathMax(window_size / 4, 3));

   out[20] = FXAI_Clamp(0.42 * ret_fast + 0.33 * ret_mid + 0.15 * ret_long + 0.10 * ret_delta, -4.0, 4.0);
   out[21] = FXAI_Clamp(0.38 * slope_m1 + 0.22 * (ret_fast - ret_mid) + 0.20 * micro_trend + 0.20 * FXAI_SharedTransferWindowFeatureSlope(x_window, window_size, 71), -4.0, 4.0);
   out[22] = FXAI_Clamp(0.34 * vol_mean + 0.22 * vol_std + 0.22 * atr_mean + 0.22 * FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 43), 0.0, 6.0);
   out[23] = FXAI_Clamp(0.26 * FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 6) +
                        0.24 * spread_shock +
                        0.18 * spread_accel +
                        0.18 * spread_log +
                        0.14 * spread_z,
                        -6.0,
                        8.0);
   out[24] = FXAI_Clamp(0.28 * ctx_mean +
                        0.14 * ctx_std +
                        0.22 * shared_util +
                        0.14 * shared_stability +
                        0.12 * shared_lead +
                        0.10 * shared_coverage,
                        -4.0,
                        4.0);
   out[25] = FXAI_Clamp(0.18 * macro_pre +
                        0.12 * macro_post +
                        0.22 * macro_imp +
                        0.18 * macro_surprise +
                        0.16 * macro_surprise_abs +
                        0.14 * macro_class,
                        -6.0,
                        6.0);
   out[26] = FXAI_Clamp(0.18 * session_transition +
                        0.14 * session_overlap +
                        0.20 * rollover_prox +
                        0.12 * triple_swap +
                        0.14 * carry_align +
                        0.12 * swap_bias +
                        0.10 * drift_mean,
                        -4.0,
                        4.0);
   out[27] = FXAI_Clamp(0.24 * drift_mean +
                        0.18 * FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 63) +
                        0.18 * FXAI_SharedTransferWindowFeatureRange(x_window, window_size, 10) +
                        0.14 * MathAbs(spread_delta) +
                        0.14 * spread_vol +
                        0.12 * MathAbs(spread_rank),
                        0.0,
                        6.0);
}

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
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      g_shared_transfer_global_b[j] = 0.0;
      g_shared_transfer_global_move[j] = 0.01 * (double)(((j * 7) % 11) - 5);
      g_shared_transfer_global_state_rec_w[j] = 0.18 + 0.02 * (double)((j % 5) - 2);
      g_shared_transfer_global_state_b[j] = 0.0;
      for(int c=0; c<3; c++)
         g_shared_transfer_global_cls[c][j] = 0.01 * (double)((((c + 1) * (j + 5)) % 9) - 4);
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
}

double FXAI_GlobalSharedTransferTrust(void)
{
   return FXAI_Clamp((double)g_shared_transfer_global_steps / 192.0, 0.0, 1.0);
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

#endif // __FXAI_TENSOR_TRANSFER_MQH__
