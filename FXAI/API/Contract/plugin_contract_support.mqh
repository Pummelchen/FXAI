class CFXAITernaryCalibrator
{
private:
   enum { CAL_BINS = 12 };

   int    m_steps;
   double m_w[3][3];
   double m_b[3];
   double m_iso_pos[3][CAL_BINS];
   double m_iso_cnt[3][CAL_BINS];

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double mx = logits[0];
      for(int c=1; c<3; c++)
         if(logits[c] > mx) mx = logits[c];

      double den = 0.0;
      for(int c=0; c<3; c++)
      {
         probs[c] = MathExp(FXAI_ClipSym(logits[c] - mx, 30.0));
         den += probs[c];
      }
      if(den <= 0.0) den = 1.0;
      for(int c=0; c<3; c++)
         probs[c] /= den;
   }

   void BuildCalLogits(const double &p_raw[], double &logits[]) const
   {
      double lraw[3];
      for(int c=0; c<3; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<3; c++)
      {
         double z = m_b[c];
         for(int j=0; j<3; j++)
            z += m_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

public:
   CFXAITernaryCalibrator(void) { Reset(); }

   void Reset(void)
   {
      m_steps = 0;
      for(int c=0; c<3; c++)
      {
         m_b[c] = 0.0;
         for(int j=0; j<3; j++)
            m_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<CAL_BINS; b++)
         {
            m_iso_pos[c][b] = 0.0;
            m_iso_cnt[c][b] = 0.0;
         }
      }
   }

   void Calibrate(const double &p_raw[], double &p_cal[]) const
   {
      double logits[3];
      BuildCalLogits(p_raw, logits);
      Softmax3(logits, p_cal);

      if(m_steps < 30)
         return;

      double p_iso[3];
      for(int c=0; c<3; c++)
      {
         double total = 0.0;
         for(int b=0; b<CAL_BINS; b++)
            total += m_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<CAL_BINS; b++)
         {
            double r = prev;
            if(m_iso_cnt[c][b] > 1e-9)
               r = m_iso_pos[c][b] / m_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= CAL_BINS) bi = CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<3; c++)
         p_cal[c] = FXAI_Clamp(0.75 * p_cal[c] + 0.25 * p_iso[c], 0.0005, 0.9990);

      double s = p_cal[0] + p_cal[1] + p_cal[2];
      if(s <= 0.0) s = 1.0;
      for(int c=0; c<3; c++)
         p_cal[c] /= s;
   }

   void Update(const double &p_raw[],
               const int cls,
               const double sample_w,
               const double lr)
   {
      double logits[3];
      BuildCalLogits(p_raw, logits);

      double p_cal[3];
      Softmax3(logits, p_cal);

      double lraw[3];
      for(int c=0; c<3; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.20, 8.00);
      double cal_lr = FXAI_Clamp(0.25 * lr * w, 0.0002, 0.0200);
      double reg_l2 = 0.0005;

      for(int c=0; c<3; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_b[c] = FXAI_ClipSym(m_b[c] + cal_lr * e, 4.0);
         for(int j=0; j<3; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double grad = e * lraw[j] - reg_l2 * (m_w[c][j] - target_w);
            m_w[c][j] = FXAI_ClipSym(m_w[c][j] + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= CAL_BINS) bi = CAL_BINS - 1;
         m_iso_cnt[c][bi] += w;
         m_iso_pos[c][bi] += w * target;
      }
      m_steps++;
   }

   int Steps(void) const { return m_steps; }

   bool Save(const int handle) const
   {
      if(handle == INVALID_HANDLE)
         return false;

      FileWriteInteger(handle, m_steps);
      for(int c=0; c<3; c++)
      {
         FileWriteDouble(handle, m_b[c]);
         for(int j=0; j<3; j++)
            FileWriteDouble(handle, m_w[c][j]);
         for(int b=0; b<CAL_BINS; b++)
         {
            FileWriteDouble(handle, m_iso_pos[c][b]);
            FileWriteDouble(handle, m_iso_cnt[c][b]);
         }
      }
      return true;
   }

   bool Load(const int handle)
   {
      if(handle == INVALID_HANDLE)
         return false;

      Reset();
      m_steps = FileReadInteger(handle);
      for(int c=0; c<3; c++)
      {
         m_b[c] = FileReadDouble(handle);
         for(int j=0; j<3; j++)
            m_w[c][j] = FileReadDouble(handle);
         for(int b=0; b<CAL_BINS; b++)
         {
            m_iso_pos[c][b] = FileReadDouble(handle);
            m_iso_cnt[c][b] = FileReadDouble(handle);
         }
      }
      return true;
   }
};

class CFXAINativeQualityHeads
{
private:
   bool   m_ready;
   int    m_steps;
   double m_w_mfe[FXAI_AI_WEIGHTS];
   double m_w_mae[FXAI_AI_WEIGHTS];
   double m_w_hit[FXAI_AI_WEIGHTS];
   double m_w_path[FXAI_AI_WEIGHTS];
   double m_w_fill[FXAI_AI_WEIGHTS];
   double m_b_mfe;
   double m_b_mae;
   double m_b_hit;
   double m_b_path;
   double m_b_fill;
   double m_w_mask[FXAI_AI_WEIGHTS];
   double m_w_vol[FXAI_AI_WEIGHTS];
   double m_w_shift[FXAI_AI_WEIGHTS];
   double m_w_ctx[FXAI_AI_WEIGHTS];
   double m_vel_mfe[FXAI_AI_WEIGHTS];
   double m_vel_mae[FXAI_AI_WEIGHTS];
   double m_vel_hit[FXAI_AI_WEIGHTS];
   double m_vel_path[FXAI_AI_WEIGHTS];
   double m_vel_fill[FXAI_AI_WEIGHTS];
   double m_vel_mask[FXAI_AI_WEIGHTS];
   double m_vel_vol[FXAI_AI_WEIGHTS];
   double m_vel_shift[FXAI_AI_WEIGHTS];
   double m_vel_ctx[FXAI_AI_WEIGHTS];
   double m_b_mask;
   double m_b_vol;
   double m_b_shift;
   double m_b_ctx;

   double Dot(const double &w[],
              const double &x[]) const
   {
      double z = w[0];
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         z += w[i] * x[i];
      return z;
   }

   void UpdateHead(double &w[],
                   double &velocity[],
                   double &bias,
                   const double &x[],
                   const double pred,
                   const double target,
                   const double lr,
                   const double l2)
   {
      double e = FXAI_Clamp(-0.5 * FXAI_LossMSEGrad(pred, target), -12.0, 12.0);
      double grad[FXAI_AI_WEIGHTS];
      ArrayInitialize(grad, 0.0);
      grad[0] = -e;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         grad[i] = -e * x[i];

      FXAIParamGroupConfig cfg = FXAI_ParamGroupMakeConfig(lr, l2, 6.0, 0.9, 0.999, 0.95, 0.25, MathMax(m_steps, 1));
      FXAIParamGroupStats stats;
      FXAI_OptSGDVectorStep(w, velocity, grad, FXAI_AI_WEIGHTS, cfg, stats);
      bias = w[0];
   }

public:
   CFXAINativeQualityHeads(void) { Reset(); }

   void Reset(void)
   {
      m_ready = false;
      m_steps = 0;
      m_b_mfe = 1.0;
      m_b_mae = -1.5;
      m_b_hit = 0.0;
      m_b_path = -0.5;
      m_b_fill = -0.5;
      m_b_mask = 0.0;
      m_b_vol = -1.0;
      m_b_shift = -1.2;
      m_b_ctx = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_w_mfe[i] = 0.0;
         m_w_mae[i] = 0.0;
         m_w_hit[i] = 0.0;
         m_w_path[i] = 0.0;
         m_w_fill[i] = 0.0;
         m_w_mask[i] = 0.0;
         m_w_vol[i] = 0.0;
         m_w_shift[i] = 0.0;
         m_w_ctx[i] = 0.0;
         m_vel_mfe[i] = 0.0;
         m_vel_mae[i] = 0.0;
         m_vel_hit[i] = 0.0;
         m_vel_path[i] = 0.0;
         m_vel_fill[i] = 0.0;
         m_vel_mask[i] = 0.0;
         m_vel_vol[i] = 0.0;
         m_vel_shift[i] = 0.0;
         m_vel_ctx[i] = 0.0;
      }
      m_w_mfe[0] = m_b_mfe;
      m_w_mae[0] = m_b_mae;
      m_w_hit[0] = m_b_hit;
      m_w_path[0] = m_b_path;
      m_w_fill[0] = m_b_fill;
      m_w_mask[0] = m_b_mask;
      m_w_vol[0] = m_b_vol;
      m_w_shift[0] = m_b_shift;
      m_w_ctx[0] = m_b_ctx;
   }

   bool Ready(void) const { return m_ready; }
   int Steps(void) const { return m_steps; }

   bool Save(const int handle) const
   {
      if(handle == INVALID_HANDLE)
         return false;

      FileWriteInteger(handle, (m_ready ? 1 : 0));
      FileWriteInteger(handle, m_steps);
      FileWriteDouble(handle, m_b_mfe);
      FileWriteDouble(handle, m_b_mae);
      FileWriteDouble(handle, m_b_hit);
      FileWriteDouble(handle, m_b_path);
      FileWriteDouble(handle, m_b_fill);
      FileWriteDouble(handle, m_b_mask);
      FileWriteDouble(handle, m_b_vol);
      FileWriteDouble(handle, m_b_shift);
      FileWriteDouble(handle, m_b_ctx);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         FileWriteDouble(handle, m_w_mfe[i]);
         FileWriteDouble(handle, m_w_mae[i]);
         FileWriteDouble(handle, m_w_hit[i]);
         FileWriteDouble(handle, m_w_path[i]);
         FileWriteDouble(handle, m_w_fill[i]);
         FileWriteDouble(handle, m_w_mask[i]);
         FileWriteDouble(handle, m_w_vol[i]);
         FileWriteDouble(handle, m_w_shift[i]);
         FileWriteDouble(handle, m_w_ctx[i]);
         FileWriteDouble(handle, m_vel_mfe[i]);
         FileWriteDouble(handle, m_vel_mae[i]);
         FileWriteDouble(handle, m_vel_hit[i]);
         FileWriteDouble(handle, m_vel_path[i]);
         FileWriteDouble(handle, m_vel_fill[i]);
         FileWriteDouble(handle, m_vel_mask[i]);
         FileWriteDouble(handle, m_vel_vol[i]);
         FileWriteDouble(handle, m_vel_shift[i]);
         FileWriteDouble(handle, m_vel_ctx[i]);
      }
      return true;
   }

   bool Load(const int handle)
   {
      if(handle == INVALID_HANDLE)
         return false;

      Reset();
      m_ready = (FileReadInteger(handle) != 0);
      m_steps = FileReadInteger(handle);
      m_b_mfe = FileReadDouble(handle);
      m_b_mae = FileReadDouble(handle);
      m_b_hit = FileReadDouble(handle);
      m_b_path = FileReadDouble(handle);
      m_b_fill = FileReadDouble(handle);
      m_b_mask = FileReadDouble(handle);
      m_b_vol = FileReadDouble(handle);
      m_b_shift = FileReadDouble(handle);
      m_b_ctx = FileReadDouble(handle);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_w_mfe[i] = FileReadDouble(handle);
         m_w_mae[i] = FileReadDouble(handle);
         m_w_hit[i] = FileReadDouble(handle);
         m_w_path[i] = FileReadDouble(handle);
         m_w_fill[i] = FileReadDouble(handle);
         m_w_mask[i] = FileReadDouble(handle);
         m_w_vol[i] = FileReadDouble(handle);
         m_w_shift[i] = FileReadDouble(handle);
         m_w_ctx[i] = FileReadDouble(handle);
         m_vel_mfe[i] = FileReadDouble(handle);
         m_vel_mae[i] = FileReadDouble(handle);
         m_vel_hit[i] = FileReadDouble(handle);
         m_vel_path[i] = FileReadDouble(handle);
         m_vel_fill[i] = FileReadDouble(handle);
         m_vel_mask[i] = FileReadDouble(handle);
         m_vel_vol[i] = FileReadDouble(handle);
         m_vel_shift[i] = FileReadDouble(handle);
         m_vel_ctx[i] = FileReadDouble(handle);
      }
      return true;
   }

   void Predict(const double &x[],
                const double move_scale,
                const double activity_gate,
                const double structural_quality,
                const double execution_quality,
                const double bank_mfe,
                const double bank_mae,
                const double bank_hit,
                const double bank_path,
                const double bank_fill,
                const double bank_trust,
                FXAIAIModelOutputV4 &out) const
   {
      double active = FXAI_Clamp(activity_gate, 0.0, 1.0);
      double structure = FXAI_Clamp(structural_quality, 0.0, 1.0);
      double exec_q = FXAI_Clamp(execution_quality, 0.0, 1.0);
      double scale = MathMax(move_scale, 0.10);
      double self_trust = FXAI_Clamp((double)m_steps / 120.0, 0.0, 0.75);
      double mix = MathMax(bank_trust, self_trust);

      double mask_self = FXAI_ClipSym(Dot(m_w_mask, x), 8.0);
      double vol_self = MathExp(FXAI_ClipSym(Dot(m_w_vol, x), 4.0));
      double shift_self = FXAI_Clamp(FXAI_Sigmoid(Dot(m_w_shift, x)), 0.0, 1.0);
      double ctx_self = FXAI_Clamp(FXAI_Sigmoid(Dot(m_w_ctx, x)), 0.0, 1.0);
      double aux_vol_ratio = FXAI_Clamp(vol_self / MathMax(scale, 0.10), 0.0, 2.0);
      structure = FXAI_Clamp(structure +
                             0.10 * (1.0 - shift_self) +
                             0.08 * ctx_self -
                             0.06 * aux_vol_ratio -
                             0.05 * FXAI_Clamp(MathAbs(mask_self) / MathMax(scale, 0.10), 0.0, 1.0),
                             0.0,
                             1.0);
      exec_q = FXAI_Clamp(exec_q +
                          0.08 * ctx_self -
                          0.10 * shift_self -
                          0.06 * aux_vol_ratio,
                          0.0,
                          1.0);

      double mfe_raw = MathLog(MathMax(scale, 0.10)) + 0.25 * active + 0.15 * structure + Dot(m_w_mfe, x);
      double mae_raw = -1.0 + 0.20 * (1.0 - structure) + 0.10 * (1.0 - exec_q) + Dot(m_w_mae, x);
      double hit_raw = 0.40 + 0.20 * active - 0.15 * structure + Dot(m_w_hit, x);
      double path_raw = -0.35 + 0.25 * (1.0 - structure) + 0.20 * (1.0 - exec_q) + Dot(m_w_path, x);
      double fill_raw = -0.50 + 0.30 * (1.0 - exec_q) + Dot(m_w_fill, x);

      double mfe_self = MathMax(out.move_q75_points, MathExp(FXAI_ClipSym(mfe_raw, 5.0)));
      double mae_self = MathMax(0.0, scale * FXAI_Sigmoid(mae_raw));
      double hit_self = FXAI_Clamp(FXAI_Sigmoid(hit_raw), 0.0, 1.0);
      double path_self = FXAI_Clamp(FXAI_Sigmoid(path_raw), 0.0, 1.0);
      double fill_self = FXAI_Clamp(FXAI_Sigmoid(fill_raw), 0.0, 1.0);

      out.mfe_mean_points = (1.0 - mix) * mfe_self + mix * MathMax(bank_mfe, out.move_q75_points);
      out.mae_mean_points = (1.0 - mix) * mae_self + mix * MathMax(0.0, bank_mae);
      out.hit_time_frac = (1.0 - mix) * hit_self + mix * FXAI_Clamp(bank_hit, 0.0, 1.0);
      out.path_risk = FXAI_Clamp((1.0 - mix) * path_self + mix * FXAI_Clamp(bank_path, 0.0, 1.0), 0.0, 1.0);
      out.fill_risk = FXAI_Clamp((1.0 - mix) * fill_self + mix * FXAI_Clamp(bank_fill, 0.0, 1.0), 0.0, 1.0);
      out.has_path_quality = true;
   }

   void Update(const double &x[],
               const double sample_w,
               const double target_mfe,
               const double target_mae,
               const double target_hit,
               const double target_path,
               const double target_fill,
               const double target_masked_step,
               const double target_next_vol,
               const double target_regime_shift,
               const double target_context_lead,
               const double lr,
               const double l2)
   {
      double w = FXAI_Clamp(sample_w, 0.10, 6.00);
      double step = FXAI_Clamp(lr * 0.35 * w, 0.0002, 0.0250);
      double reg = FXAI_Clamp(l2, 0.0, 0.05);

      double pred_mfe = MathExp(FXAI_ClipSym(MathLog(0.10) + Dot(m_w_mfe, x), 5.0));
      double pred_mae = FXAI_Sigmoid(Dot(m_w_mae, x));
      double pred_hit = FXAI_Sigmoid(Dot(m_w_hit, x));
      double pred_path = FXAI_Sigmoid(Dot(m_w_path, x));
      double pred_fill = FXAI_Sigmoid(Dot(m_w_fill, x));
      double pred_mask = FXAI_ClipSym(Dot(m_w_mask, x), 8.0);
      double pred_vol = MathExp(FXAI_ClipSym(Dot(m_w_vol, x), 4.0));
      double pred_shift = FXAI_Sigmoid(Dot(m_w_shift, x));
      double pred_ctx = FXAI_Sigmoid(Dot(m_w_ctx, x));

      double tgt_mfe = MathLog(MathMax(target_mfe, 0.10));
      double tgt_mae = FXAI_Clamp(target_mae, 0.0, 1.0);
      double tgt_hit = FXAI_Clamp(target_hit, 0.0, 1.0);
      double tgt_path = FXAI_Clamp(target_path, 0.0, 1.0);
      double tgt_fill = FXAI_Clamp(target_fill, 0.0, 1.0);
      double tgt_mask = FXAI_ClipSym(target_masked_step, 8.0);
      double tgt_vol = MathLog(MathMax(target_next_vol, 0.05));
      double tgt_shift = FXAI_Clamp(target_regime_shift, 0.0, 1.0);
      double tgt_ctx = FXAI_Clamp(target_context_lead, 0.0, 1.0);

      UpdateHead(m_w_mfe, m_vel_mfe, m_b_mfe, x, MathLog(MathMax(pred_mfe, 0.10)), tgt_mfe, step, reg);
      UpdateHead(m_w_mae, m_vel_mae, m_b_mae, x, pred_mae, tgt_mae, step, reg);
      UpdateHead(m_w_hit, m_vel_hit, m_b_hit, x, pred_hit, tgt_hit, step, reg);
      UpdateHead(m_w_path, m_vel_path, m_b_path, x, pred_path, tgt_path, step, reg);
      UpdateHead(m_w_fill, m_vel_fill, m_b_fill, x, pred_fill, tgt_fill, step, reg);
      UpdateHead(m_w_mask, m_vel_mask, m_b_mask, x, pred_mask, tgt_mask, 0.80 * step, reg);
      UpdateHead(m_w_vol, m_vel_vol, m_b_vol, x, MathLog(MathMax(pred_vol, 0.05)), tgt_vol, 0.70 * step, reg);
      UpdateHead(m_w_shift, m_vel_shift, m_b_shift, x, pred_shift, tgt_shift, 0.75 * step, reg);
      UpdateHead(m_w_ctx, m_vel_ctx, m_b_ctx, x, pred_ctx, tgt_ctx, 0.75 * step, reg);
      m_ready = true;
      m_steps++;
   }
};
