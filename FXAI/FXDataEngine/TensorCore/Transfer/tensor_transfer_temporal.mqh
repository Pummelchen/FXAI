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
   double macro_state_quality = FXAI_SharedTransferWindowFeatureAt(x_window,
                                                                   window_size,
                                                                   bar_idx,
                                                                   FXAI_MACRO_EVENT_FEATURE_OFFSET + 19,
                                                                   0.0);
   double macro_policy_pressure = FXAI_SharedTransferWindowFeatureAt(x_window,
                                                                     window_size,
                                                                     bar_idx,
                                                                     FXAI_MACRO_EVENT_FEATURE_OFFSET + 15,
                                                                     0.0);
   double macro_policy_div = FXAI_SharedTransferWindowFeatureAt(x_window,
                                                                window_size,
                                                                bar_idx,
                                                                FXAI_MACRO_EVENT_FEATURE_OFFSET + 14,
                                                                0.0);
   bar_feats[10] = FXAI_Clamp(0.70 * bar_feats[10] + 0.30 * macro_state_quality, 0.0, 1.0);
   bar_feats[11] = FXAI_Clamp(0.50 * FXAI_SharedTransferWindowFeatureAt(x_window,
                                                                        window_size,
                                                                        bar_idx,
                                                                        FXAI_MACRO_EVENT_FEATURE_OFFSET + 3,
                                                                        0.0) +
                              0.30 * macro_policy_pressure +
                              0.20 * macro_policy_div,
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
