   void EncodeSharedTransferBackbone(const double &a[],
                                     double &latent[]) const
   {
      double seq_tokens[];
      FXAI_SharedTransferBuildSequenceTokens(a, m_ctx_window, m_ctx_window_size, seq_tokens);
      FXAI_SharedTransferEncodeTemporal(a,
                                        seq_tokens,
                                        m_ctx_window,
                                        m_ctx_window_size,
                                        FXAI_SharedTransferDomainBucket(m_ctx_domain_hash),
                                        FXAI_SharedTransferHorizonBucket(m_ctx_horizon_minutes),
                                        m_ctx_session_bucket,
                                        m_shared_backbone_w,
                                        m_shared_backbone_seq_w,
                                        m_shared_backbone_time_w,
                                        m_shared_backbone_time_gate_w,
                                        m_shared_backbone_state_w,
                                        m_shared_backbone_state_rec_w,
                                        m_shared_backbone_state_b,
                                        m_shared_backbone_b,
                                        m_shared_domain_emb,
                                        m_shared_horizon_emb,
                                        m_shared_session_emb,
                                        latent);
   }

   void PredictSharedTransferBackbone(const double &a[],
                                      double &probs[],
                                      double &move_adj) const
   {
      double latent[];
      EncodeSharedTransferBackbone(a, latent);

      double logits[3];
      for(int c=0; c<3; c++)
      {
         logits[c] = 0.0;
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            logits[c] += m_shared_backbone_cls[c][j] * latent[j];
      }
      FXAI_SharedTransferSoftmax(logits, probs);

      move_adj = 0.0;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         move_adj += m_shared_backbone_move[j] * latent[j];
   }

   double BuildTransferSlotSignal(const double &x[],
                                  const int slot) const
   {
      if(slot < 0 || slot >= FXAI_CONTEXT_TOP_SYMBOLS)
         return 0.0;
      int base = 50 + slot * 4;
      double ctx_ret = FXAI_GetInputFeature(x, base + 0);
      double ctx_lag = FXAI_GetInputFeature(x, base + 1);
      double ctx_rel = FXAI_GetInputFeature(x, base + 2);
      double signal = 0.30 * ctx_ret + 0.50 * ctx_lag + 0.20 * ctx_rel;
      return FXAI_ClipSym(signal, 4.0);
   }

   void BlendTransferSlotPriors(const double &x[],
                                double &probs[],
                                double &move_scale_mult,
                                double &reliability_boost) const
   {
      double coverage = FXAI_Clamp(0.5 + 0.5 * FXAI_GetInputFeature(x, 65), 0.0, 1.0);
      double domain_buy = 0.0;
      double domain_sell = 0.0;
      double domain_skip = 0.0;
      double domain_move = 0.0;
      double domain_rel = 0.0;
      double domain_weight = 0.0;
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         if(m_transfer_slot_obs[slot] <= 1e-6)
            continue;

         int base = 50 + slot * 4;
         double ctx_corr = FXAI_GetInputFeature(x, base + 3);
         double signal = BuildTransferSlotSignal(x, slot);
         double obs_trust = FXAI_Clamp(m_transfer_slot_obs[slot] / 24.0, 0.0, 1.0);
         double w = FXAI_Clamp(obs_trust *
                               (0.25 + 0.75 * MathAbs(ctx_corr)) *
                               (0.20 + 0.80 * coverage) *
                               (0.20 + 0.80 * MathAbs(signal)),
                               0.0,
                               2.0);
         if(w <= 1e-6)
            continue;

         double align = FXAI_Clamp(m_transfer_slot_align[slot], -1.0, 1.0);
         double lead = FXAI_Clamp(m_transfer_slot_lead[slot], 0.0, 1.0);
         double move_scale = FXAI_Clamp(m_transfer_slot_move[slot], 0.50, 2.50);
         double buy_prior = 0.10;
         double sell_prior = 0.10;
         double skip_prior = FXAI_Clamp(0.55 - 0.10 * MathAbs(signal), 0.05, 0.80);
         if(signal > 0.0)
         {
            buy_prior = FXAI_Clamp(0.45 + 0.22 * align + 0.15 * lead + 0.08 * MathAbs(signal), 0.05, 0.95);
            sell_prior = FXAI_Clamp(0.20 - 0.12 * align, 0.02, 0.60);
         }
         else if(signal < 0.0)
         {
            sell_prior = FXAI_Clamp(0.45 - 0.22 * align + 0.15 * lead + 0.08 * MathAbs(signal), 0.05, 0.95);
            buy_prior = FXAI_Clamp(0.20 + 0.12 * align, 0.02, 0.60);
         }

         double ps = buy_prior + sell_prior + skip_prior;
         if(ps <= 0.0) ps = 1.0;
         buy_prior /= ps;
         sell_prior /= ps;
         skip_prior /= ps;

         domain_buy += w * buy_prior;
         domain_sell += w * sell_prior;
         domain_skip += w * skip_prior;
         domain_move += w * move_scale;
         domain_rel += w * (0.50 + 0.25 * MathAbs(align) + 0.25 * lead);
         domain_weight += w;
      }

      if(domain_weight <= 1e-6)
      {
         move_scale_mult = 1.0;
         reliability_boost = 0.0;
         return;
      }

      probs[0] = domain_sell / domain_weight;
      probs[1] = domain_buy / domain_weight;
      probs[2] = domain_skip / domain_weight;
      move_scale_mult = FXAI_Clamp(domain_move / domain_weight, 0.70, 1.50);
      reliability_boost = FXAI_Clamp((domain_rel / domain_weight) - 0.50, -0.15, 0.20);
   }

   void ApplySharedContextAdapter(FXAIAIModelOutputV4 &out,
                                  const double &x[]) const
   {
      double a[];
      BuildSharedAdapterInput(x, a);
      if(!HasSharedAdapterSignal(a))
         return;

      double shallow_trust = FXAI_Clamp((double)m_shared_adapter_steps / 96.0, 0.0, 1.0);
      shallow_trust *= SharedAdapterSignalStrength(a);
      double backbone_trust = FXAI_Clamp((double)m_shared_backbone_steps / 144.0, 0.0, 1.0);
      backbone_trust *= FXAI_Clamp(0.10 + 1.15 * SharedAdapterSignalStrength(a), 0.0, 0.55);
      double trust = MathMax(shallow_trust, backbone_trust);
      if(trust <= 1e-6)
         return;

      double logits[3];
      for(int c=0; c<3; c++)
      {
         double z = 0.0;
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            z += m_shared_cls_w[c][i] * a[i];
         logits[c] = z;
      }
      double probs[3];
      FXAI_SharedTransferSoftmax(logits, probs);

      double bb_probs[];
      double bb_move_adj = 0.0;
      PredictSharedTransferBackbone(a, bb_probs, bb_move_adj);
      if(ArraySize(bb_probs) == 3 && backbone_trust > 1e-6)
      {
         double blend = FXAI_Clamp(0.35 + 0.65 * backbone_trust, 0.0, 0.85);
         for(int c=0; c<3; c++)
            probs[c] = FXAI_Clamp((1.0 - blend) * probs[c] + blend * bb_probs[c], 0.0005, 0.9990);
      }

      double global_probs[];
      double global_move_adj = 0.0;
      FXAI_GlobalSharedTransferPredict(a,
                                       m_ctx_window,
                                       m_ctx_window_size,
                                       m_ctx_domain_hash,
                                       m_ctx_horizon_minutes,
                                       m_ctx_session_bucket,
                                       global_probs,
                                       global_move_adj);
      double global_trust = FXAI_GlobalSharedTransferTrust();
      global_trust *= FXAI_Clamp(0.12 + 1.10 * SharedAdapterSignalStrength(a), 0.0, 0.65);
      if(ArraySize(global_probs) == 3 && global_trust > 1e-6)
      {
         double blend = FXAI_Clamp(0.30 + 0.60 * global_trust, 0.0, 0.78);
         for(int c=0; c<3; c++)
            probs[c] = FXAI_Clamp((1.0 - blend) * probs[c] + blend * global_probs[c], 0.0005, 0.9990);
      }

      FXAIFoundationSignals foundation;
      FXAI_GlobalFoundationPredict(a,
                                   m_ctx_window,
                                   m_ctx_window_size,
                                   m_ctx_domain_hash,
                                   m_ctx_horizon_minutes,
                                   m_ctx_session_bucket,
                                   foundation);
      double foundation_trust = foundation.trust *
                                FXAI_Clamp(0.10 + 1.10 * SharedAdapterSignalStrength(a), 0.0, 0.60);
      if(foundation_trust > 1e-6)
      {
         double dir_hint = FXAI_Clamp(foundation.direction_bias, -1.0, 1.0);
         probs[(int)FXAI_LABEL_BUY] = FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] + 0.08 * foundation_trust * MathMax(dir_hint, 0.0), 0.0005, 0.9990);
         probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(probs[(int)FXAI_LABEL_SELL] + 0.08 * foundation_trust * MathMax(-dir_hint, 0.0), 0.0005, 0.9990);
         probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(probs[(int)FXAI_LABEL_SKIP] - 0.06 * foundation_trust * foundation.tradability, 0.0005, 0.9990);
      }

      FXAIStudentSignals student;
      FXAI_GlobalStudentPredict(a,
                                m_ctx_window,
                                m_ctx_window_size,
                                m_ctx_domain_hash,
                                m_ctx_horizon_minutes,
                                m_ctx_session_bucket,
                                student);
      double student_trust = student.trust *
                             FXAI_Clamp(0.10 + 1.00 * SharedAdapterSignalStrength(a), 0.0, 0.62);
      if(student_trust > 1e-6)
      {
         double blend = FXAI_Clamp(0.24 + 0.52 * student_trust, 0.0, 0.70);
         for(int c=0; c<3; c++)
            probs[c] = FXAI_Clamp((1.0 - blend) * probs[c] + blend * student.class_probs[c], 0.0005, 0.9990);
      }

      FXAIAnalogMemoryQuery analog;
      FXAI_QueryAnalogMemory(x,
                             m_ctx_regime_id,
                             m_ctx_session_bucket,
                             m_ctx_horizon_minutes,
                             m_ctx_domain_hash,
                             analog);
      double analog_trust = FXAI_Clamp(0.50 * analog.similarity + 0.25 * analog.quality + 0.10 * analog.domain_alignment, 0.0, 0.40);
      if(analog_trust > 1e-6)
      {
         double analog_dir = FXAI_Clamp(analog.direction_agreement, -1.0, 1.0);
         probs[(int)FXAI_LABEL_BUY] = FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] + analog_trust * MathMax(analog_dir, 0.0), 0.0005, 0.9990);
         probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(probs[(int)FXAI_LABEL_SELL] + analog_trust * MathMax(-analog_dir, 0.0), 0.0005, 0.9990);
         probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(probs[(int)FXAI_LABEL_SKIP] - 0.08 * analog_trust * analog.quality, 0.0005, 0.9990);
      }

      double transfer_probs[3];
      transfer_probs[0] = 0.0;
      transfer_probs[1] = 0.0;
      transfer_probs[2] = 0.0;
      double transfer_move_mult = 1.0;
      double transfer_rel_boost = 0.0;
      BlendTransferSlotPriors(x, transfer_probs, transfer_move_mult, transfer_rel_boost);
      double transfer_mass = transfer_probs[0] + transfer_probs[1] + transfer_probs[2];
      if(transfer_mass > 0.0)
      {
         for(int c=0; c<3; c++)
            transfer_probs[c] /= transfer_mass;
         double transfer_trust = FXAI_Clamp(0.08 + 0.28 * a[4] + 0.08 * MathAbs(a[24]) + 0.06 * a[22], 0.0, 0.34);
         for(int c=0; c<3; c++)
            probs[c] = FXAI_Clamp((1.0 - transfer_trust) * probs[c] + transfer_trust * transfer_probs[c], 0.0005, 0.9990);
      }

      for(int c=0; c<3; c++)
         out.class_probs[c] = FXAI_Clamp((1.0 - trust) * out.class_probs[c] + trust * probs[c], 0.0005, 0.9990);
      NormalizeClassDistribution(out.class_probs);

      double move_adj = 0.0;
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         move_adj += m_shared_move_w[i] * a[i];
      move_adj = 0.55 * move_adj + 0.20 * bb_move_adj + 0.25 * global_move_adj;
      double scale = FXAI_Clamp((1.0 + 0.16 * trust * FXAI_ClipSym(move_adj, 1.5)) *
                                transfer_move_mult *
                                FXAI_Clamp(0.90 +
                                           0.06 * foundation_trust * foundation.move_ratio +
                                           0.05 * student_trust * student.move_ratio +
                                           0.04 * analog_trust * (1.0 + analog.edge_norm),
                                           0.75,
                                           1.50),
                                0.75,
                                1.55);
      out.move_mean_points = MathMax(0.0, out.move_mean_points * scale);
      out.move_q25_points = MathMax(0.0, out.move_q25_points * scale);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_q50_points * scale);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_q75_points * scale);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY],
                                          out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(out.reliability * (1.0 - 0.12 * trust) +
                                   trust * FXAI_Clamp(0.48 +
                                                      0.14 * a[2] +
                                                      0.10 * a[4] +
                                                      0.05 * MathAbs(a[11]) +
                                                      0.05 * MathAbs(a[15]) +
                                                      0.04 * (1.0 - MathAbs(a[19]) / 6.0) +
                                                      0.04 * a[22] +
                                                      0.04 * MathAbs(a[24]) +
                                                      0.03 * (1.0 - FXAI_Clamp(a[27] / 6.0, 0.0, 1.0)) +
                                                      0.04 * foundation_trust * foundation.tradability +
                                                      0.03 * student_trust * student.tradability +
                                                      0.03 * analog_trust * analog.quality +
                                                      transfer_rel_boost,
                                                      0.0,
                                                      1.0),
                                   0.0,
                                   1.0);
      out.path_risk = FXAI_Clamp(out.path_risk * (1.0 - 0.08 * analog_trust) +
                                 analog_trust * (1.0 - analog.path_safety),
                                 0.0,
                                 1.0);
      out.fill_risk = FXAI_Clamp(out.fill_risk * (1.0 - 0.08 * analog_trust) +
                                 analog_trust * (1.0 - analog.execution_safety),
                                 0.0,
                                 1.0);
   }

   void UpdateCrossSymbolTransferBank(const double &x[],
                                      const double move_points,
                                      const double sample_w)
   {
      double move_sign = FXAI_Sign(move_points);
      if(MathAbs(move_sign) <= 1e-9)
         return;

      double coverage = FXAI_Clamp(0.5 + 0.5 * FXAI_GetInputFeature(x, 65), 0.0, 1.0);
      double move_scale = MathMax(MathAbs(move_points), MathMax(ResolveMinMovePoints(), 0.10));
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         int base = 50 + slot * 4;
         double ctx_corr = FXAI_GetInputFeature(x, base + 3);
         double signal = BuildTransferSlotSignal(x, slot);
         if(MathAbs(signal) <= 1e-6)
            continue;

         double trust = FXAI_Clamp((0.30 + 0.70 * MathAbs(ctx_corr)) *
                                   (0.20 + 0.80 * coverage) *
                                   FXAI_Clamp(sample_w, 0.25, 4.0),
                                   0.02,
                                   2.50);
         double alpha = FXAI_Clamp(0.05 * trust / MathSqrt(1.0 + 0.02 * m_transfer_slot_obs[slot]), 0.01, 0.20);
         double align_target = FXAI_Clamp(FXAI_Sign(signal) * move_sign, -1.0, 1.0);
         double lead_target = FXAI_Clamp(0.5 + 0.5 * FXAI_Sign(FXAI_GetInputFeature(x, base + 1)) * move_sign, 0.0, 1.0);
         double move_target = FXAI_Clamp(move_scale / MathMax(MathAbs(signal), 0.10), 0.50, 2.50);

         if(m_transfer_slot_obs[slot] <= 1e-6)
         {
            m_transfer_slot_align[slot] = align_target;
            m_transfer_slot_lead[slot] = lead_target;
            m_transfer_slot_move[slot] = move_target;
         }
         else
         {
            m_transfer_slot_align[slot] = FXAI_Clamp((1.0 - alpha) * m_transfer_slot_align[slot] + alpha * align_target, -1.0, 1.0);
            m_transfer_slot_lead[slot] = FXAI_Clamp((1.0 - alpha) * m_transfer_slot_lead[slot] + alpha * lead_target, 0.0, 1.0);
            m_transfer_slot_move[slot] = FXAI_Clamp((1.0 - alpha) * m_transfer_slot_move[slot] + alpha * move_target, 0.50, 2.50);
         }
         m_transfer_slot_obs[slot] = MathMin(m_transfer_slot_obs[slot] + trust, 5000.0);
      }
   }

   void UpdateSharedContextAdapter(const double &x[],
                                   const int y,
                                   const double move_points,
                                   const double sample_w,
                                   const double lr)
   {
      double a[];
      BuildSharedAdapterInput(x, a);
      if(!HasSharedAdapterSignal(a))
         return;

      int cls = NormalizeClassLabel(y, x, move_points);
      double logits[3];
      for(int c=0; c<3; c++)
      {
         double z = 0.0;
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            z += m_shared_cls_w[c][i] * a[i];
         logits[c] = z;
      }

      double probs[3];
      FXAI_SharedTransferSoftmax(logits, probs);

      double step = FXAI_Clamp(0.18 * lr * FXAI_Clamp(sample_w, 0.25, 4.0), 0.0002, 0.0200);
      for(int c=0; c<3; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double err = target - probs[c];
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            m_shared_cls_w[c][i] = FXAI_ClipSym(m_shared_cls_w[c][i] + step * err * a[i], 3.0);
      }

      double move_pred = 0.0;
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         move_pred += m_shared_move_w[i] * a[i];
      double move_target = FXAI_Clamp(MathLog(1.0 + MathAbs(move_points)), 0.0, 4.0);
      double move_err = FXAI_ClipSym(move_target - move_pred, 3.0);
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         m_shared_move_w[i] = FXAI_ClipSym(m_shared_move_w[i] + 0.80 * step * move_err * a[i], 3.0);

      double latent[];
      double seq_tokens[];
      FXAI_SharedTransferBuildSequenceTokens(a, m_ctx_window, m_ctx_window_size, seq_tokens);
      EncodeSharedTransferBackbone(a, latent);
      double bb_logits[3];
      for(int c=0; c<3; c++)
      {
         bb_logits[c] = 0.0;
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            bb_logits[c] += m_shared_backbone_cls[c][j] * latent[j];
      }
      double bb_probs[];
      FXAI_SharedTransferSoftmax(bb_logits, bb_probs);

      double latent_grad[FXAI_SHARED_TRANSFER_LATENT];
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         latent_grad[j] = 0.0;

      double bb_step = FXAI_Clamp(0.12 * lr * FXAI_Clamp(sample_w, 0.25, 4.0), 0.0001, 0.0120);
      for(int c=0; c<3; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double err = target - (ArraySize(bb_probs) == 3 ? bb_probs[c] : 0.3333333);
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         {
            latent_grad[j] += err * m_shared_backbone_cls[c][j];
            m_shared_backbone_cls[c][j] = FXAI_ClipSym(m_shared_backbone_cls[c][j] + bb_step * err * latent[j], 3.0);
         }
      }

      double bb_move_pred = 0.0;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         bb_move_pred += m_shared_backbone_move[j] * latent[j];
      double bb_move_err = FXAI_ClipSym(move_target - bb_move_pred, 3.0);
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      {
         latent_grad[j] += 0.30 * bb_move_err * m_shared_backbone_move[j];
         m_shared_backbone_move[j] = FXAI_ClipSym(m_shared_backbone_move[j] + 0.65 * bb_step * bb_move_err * latent[j], 3.0);
      }

      int domain_bucket = FXAI_SharedTransferDomainBucket(m_ctx_domain_hash);
      int horizon_bucket = FXAI_SharedTransferHorizonBucket(m_ctx_horizon_minutes);
      int session_bucket = m_ctx_session_bucket;
      if(session_bucket < 0) session_bucket = 0;
      if(session_bucket >= FXAI_PLUGIN_SESSION_BUCKETS) session_bucket = FXAI_PLUGIN_SESSION_BUCKETS - 1;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      {
         double g = FXAI_ClipSym(latent_grad[j] * (1.0 - latent[j] * latent[j]), 2.5);
         m_shared_backbone_b[j] = FXAI_ClipSym(m_shared_backbone_b[j] + bb_step * g, 3.0);
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            m_shared_backbone_w[j][i] = FXAI_ClipSym(m_shared_backbone_w[j][i] + bb_step * g * a[i], 3.0);
         for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
            m_shared_backbone_seq_w[j][t] = FXAI_ClipSym(m_shared_backbone_seq_w[j][t] + 0.80 * bb_step * g * FXAI_SharedTransferArrayValue(seq_tokens, t, 0.0), 3.0);
         if(m_ctx_window_size > 0)
         {
            double pooled_den = 0.0;
            double pooled_val = FXAI_SharedTransferTemporalPoolLatent(m_ctx_window,
                                                                      m_ctx_window_size,
                                                                      m_shared_backbone_time_w,
                                                                      m_shared_backbone_time_gate_w,
                                                                      j);
            int bars = MathMin(m_ctx_window_size, FXAI_MAX_SEQUENCE_BARS);
            for(int b=0; b<bars; b++)
            {
               double bar_feats[];
               FXAI_SharedTransferExtractBarFeatures(m_ctx_window, m_ctx_window_size, b, bar_feats);
               double recency = 1.0 / (1.0 + 0.08 * (double)b);
               double recency_pos = 1.0 - ((double)b / (double)MathMax(bars - 1, 1));
               double gate = recency * FXAI_SharedTransferTemporalGateAt(m_shared_backbone_time_gate_w, j, bar_feats, recency_pos);
               pooled_den += gate;
            }
            if(pooled_den > 1e-6)
            {
               for(int b=0; b<bars; b++)
               {
                  double bar_feats[];
                  FXAI_SharedTransferExtractBarFeatures(m_ctx_window, m_ctx_window_size, b, bar_feats);
                  double recency = 1.0 / (1.0 + 0.08 * (double)b);
                  double recency_pos = 1.0 - ((double)b / (double)MathMax(bars - 1, 1));
                  double gate = recency * FXAI_SharedTransferTemporalGateAt(m_shared_backbone_time_gate_w, j, bar_feats, recency_pos);
                  if(gate <= 1e-6)
                     continue;
                  double norm_gate = gate / pooled_den;
                  double bar_val = FXAI_SharedTransferTemporalValueAt(m_shared_backbone_time_w, j, bar_feats);
                  for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
                  {
                     double feat_v = FXAI_SharedTransferArrayValue(bar_feats, c, 0.0);
                     m_shared_backbone_time_w[j][c] =
                        FXAI_ClipSym(m_shared_backbone_time_w[j][c] + 0.55 * bb_step * g * norm_gate * feat_v, 3.0);
                     m_shared_backbone_time_gate_w[j][c] =
                        FXAI_ClipSym(m_shared_backbone_time_gate_w[j][c] +
                                     0.16 * bb_step * g * norm_gate * (bar_val - pooled_val) * feat_v,
                                     3.0);
                  }
               }
            }
         }
         double state_last = 0.0;
         double state_mean = 0.0;
         double state_abs = 0.0;
         FXAI_SharedTransferTemporalStateSummary(m_ctx_window,
                                                 m_ctx_window_size,
                                                 m_shared_backbone_state_w,
                                                 m_shared_backbone_state_rec_w,
                                                 m_shared_backbone_state_b,
                                                 j,
                                                 state_last,
                                                 state_mean,
                                                 state_abs);
         for(int c=0; c<FXAI_SHARED_TRANSFER_STATE_FEATURES; c++)
         {
            double feat_mean = FXAI_SharedTransferWindowFeatureMean(m_ctx_window, m_ctx_window_size, c);
            m_shared_backbone_state_w[j][c] =
               FXAI_ClipSym(m_shared_backbone_state_w[j][c] +
                            0.36 * bb_step * g * FXAI_ClipSym(feat_mean, 4.0),
                            3.0);
         }
         m_shared_backbone_state_rec_w[j] =
            FXAI_ClipSym(m_shared_backbone_state_rec_w[j] +
                         0.20 * bb_step * g * FXAI_ClipSym(state_mean + 0.35 * state_last, 2.5),
                         2.5);
         m_shared_backbone_state_b[j] =
            FXAI_ClipSym(m_shared_backbone_state_b[j] +
                         0.25 * bb_step * g * (0.35 + 0.65 * state_abs),
                         3.0);
         m_shared_domain_emb[domain_bucket][j] = FXAI_ClipSym(m_shared_domain_emb[domain_bucket][j] + 0.40 * bb_step * g, 3.0);
         m_shared_horizon_emb[horizon_bucket][j] = FXAI_ClipSym(m_shared_horizon_emb[horizon_bucket][j] + 0.40 * bb_step * g, 3.0);
         m_shared_session_emb[session_bucket][j] = FXAI_ClipSym(m_shared_session_emb[session_bucket][j] + 0.30 * bb_step * g, 3.0);
      }

      m_shared_adapter_steps++;
      if(m_shared_adapter_steps >= 24)
         m_shared_adapter_ready = true;
      m_shared_backbone_steps++;
      if(m_shared_backbone_steps >= 36)
         m_shared_backbone_ready = true;

      FXAI_GlobalSharedTransferUpdate(a,
                                      m_ctx_window,
                                      m_ctx_window_size,
                                      m_ctx_domain_hash,
                                      m_ctx_horizon_minutes,
                                      m_ctx_session_bucket,
                                      cls,
                                      m_ctx_cost_points,
                                      move_points,
                                      sample_w,
                                      lr);
   }

