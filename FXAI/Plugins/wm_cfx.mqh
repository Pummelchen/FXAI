#ifndef __FXAI_AI_CFX_WORLD_MQH__
#define __FXAI_AI_CFX_WORLD_MQH__

#include "..\API\plugin_base.mqh"

#define FXAI_CFXW_STATE   6
#define FXAI_CFXW_FEATS   8
#define FXAI_CFXW_PARTS   16
#define FXAI_CFXW_HORIZON 5

class CFXAIAICFXWorld : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_steps;
   uint   m_seed;
   double m_state[FXAI_CFXW_STATE];
   double m_state_w[FXAI_CFXW_STATE][FXAI_CFXW_FEATS];
   double m_state_g2[FXAI_CFXW_STATE][FXAI_CFXW_FEATS];
   double m_trans[FXAI_CFXW_STATE];
   double m_confidence_ema;
   double m_consensus_ema;
   double m_edge_ema;
   double m_downside_ema;
   double m_session_edge[4];
   CFXAITernaryCalibrator m_cal3;

   double ClampProb(const double p) const
   {
      return FXAI_Clamp(p, 0.0005, 0.9995);
   }

   void ResetModel(void)
   {
      m_initialized = false;
      m_steps = 0;
      m_seed = (uint)2166136261;
      m_cal3.Reset();
      for(int s=0; s<FXAI_CFXW_STATE; s++)
      {
         m_state[s] = 0.0;
         m_trans[s] = 0.0;
         for(int k=0; k<FXAI_CFXW_FEATS; k++)
         {
            m_state_w[s][k] = 0.0;
            m_state_g2[s][k] = 0.0;
         }
      }
      m_confidence_ema = 0.50;
      m_consensus_ema  = 0.50;
      m_edge_ema       = 0.0;
      m_downside_ema   = 0.0;
      for(int b=0; b<4; b++) m_session_edge[b] = 0.0;
   }

   void SeedModel(void)
   {
      // State order:
      // 0 drift, 1 mean-reversion, 2 volatility, 3 breakout_hazard, 4 liquidity_noise, 5 confidence_memory
      m_trans[0] = 0.72;
      m_trans[1] = 0.58;
      m_trans[2] = 0.82;
      m_trans[3] = 0.60;
      m_trans[4] = 0.78;
      m_trans[5] = 0.90;

      // Small hand-crafted priors before online adaptation.
      m_state_w[0][0] =  0.35; m_state_w[0][1] =  0.24; m_state_w[0][2] = -0.08; m_state_w[0][3] =  0.10;
      m_state_w[0][4] =  0.06; m_state_w[0][5] =  0.04; m_state_w[0][6] = -0.12; m_state_w[0][7] =  0.05;

      m_state_w[1][0] = -0.25; m_state_w[1][1] =  0.08; m_state_w[1][2] =  0.28; m_state_w[1][3] = -0.06;
      m_state_w[1][4] =  0.04; m_state_w[1][5] =  0.10; m_state_w[1][6] =  0.06; m_state_w[1][7] = -0.04;

      m_state_w[2][0] =  0.04; m_state_w[2][1] =  0.38; m_state_w[2][2] =  0.06; m_state_w[2][3] =  0.18;
      m_state_w[2][4] =  0.28; m_state_w[2][5] =  0.04; m_state_w[2][6] =  0.08; m_state_w[2][7] =  0.10;

      m_state_w[3][0] =  0.12; m_state_w[3][1] =  0.26; m_state_w[3][2] = -0.12; m_state_w[3][3] =  0.32;
      m_state_w[3][4] = -0.08; m_state_w[3][5] =  0.05; m_state_w[3][6] = -0.06; m_state_w[3][7] =  0.06;

      m_state_w[4][0] =  0.02; m_state_w[4][1] =  0.20; m_state_w[4][2] =  0.10; m_state_w[4][3] =  0.06;
      m_state_w[4][4] =  0.36; m_state_w[4][5] =  0.10; m_state_w[4][6] =  0.18; m_state_w[4][7] =  0.04;

      m_state_w[5][0] =  0.16; m_state_w[5][1] = -0.10; m_state_w[5][2] = -0.06; m_state_w[5][3] =  0.12;
      m_state_w[5][4] = -0.14; m_state_w[5][5] =  0.08; m_state_w[5][6] = -0.12; m_state_w[5][7] =  0.06;

      m_initialized = true;
   }

   void EnsureBootstrapped(void)
   {
      if(m_initialized) return;
      ResetModel();
      SeedModel();
   }

   double SafeX(const double &x[], const int idx) const
   {
      if(idx < 0 || idx >= ArraySize(x)) return 0.0;
      double v = x[idx];
      if(!MathIsValidNumber(v)) return 0.0;
      return FXAI_ClipSym(v, 8.0);
   }

   int SessionBucket(void) const
   {
      MqlDateTime dt;
      TimeToStruct(ResolveContextTime(), dt);
      if(dt.hour < 6) return 0;
      if(dt.hour < 12) return 1;
      if(dt.hour < 17) return 2;
      return 3;
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         xa[i] = (i < xn ? SafeX(x, i) : 0.0);

      int win_n = CurrentWindowSize();
      if(win_n <= 1) return;

      double mean1 = CurrentWindowFeatureMean(0);
      double mean2 = CurrentWindowFeatureMean(1);
      double mean6 = CurrentWindowFeatureMean(5);
      double mean7 = CurrentWindowFeatureMean(6);
      double first1 = CurrentWindowValue(0, 1);
      double last1  = CurrentWindowValue(win_n - 1, 1);
      double drift = last1 - first1;

      xa[1] = FXAI_ClipSym(0.55 * xa[1] + 0.25 * mean1 + 0.20 * drift, 8.0);
      xa[2] = FXAI_ClipSym(0.65 * xa[2] + 0.35 * mean6, 8.0);
      xa[5] = FXAI_ClipSym(0.70 * xa[5] + 0.30 * mean2, 8.0);
      xa[7] = FXAI_ClipSym(0.65 * xa[7] + 0.35 * mean7, 8.0);
   }

   int DynamicParticleCount(const double &f[], const double cost_points, const double min_move_points) const
   {
      double stress = MathAbs(f[2]) + 0.7 * MathAbs(f[4]) + 0.6 * MathAbs(f[5]);
      double cost_ratio = cost_points / MathMax(min_move_points, 0.10);
      int n = FXAI_CFXW_PARTS + (int)MathRound(3.0 * FXAI_Clamp(stress, 0.0, 2.5)) - (int)MathRound(FXAI_Clamp(cost_ratio - 1.0, 0.0, 3.0));
      if(SessionBucket() == 2) n += 2;
      if(SessionBucket() == 0) n += 1;
      if(n < 12) n = 12;
      if(n > 28) n = 28;
      return n;
   }

   int DynamicHorizonSteps(const double &f[]) const
   {
      double regime_drive = MathAbs(f[1]) + 0.7 * MathAbs(f[3]) + 0.3 * MathAbs(f[6]);
      int h = FXAI_CFXW_HORIZON + (int)MathRound(FXAI_Clamp(regime_drive - 0.8, 0.0, 3.0));
      if(SessionBucket() == 0) h -= 1;
      if(h < 4) h = 4;
      if(h > 9) h = 9;
      return h;
   }

   double DynamicNoiseScale(const double &f[]) const
   {
      double vol = 0.65 + 0.25 * MathAbs(f[2]) + 0.15 * MathAbs(f[5]);
      int bucket = SessionBucket();
      if(bucket == 0) vol *= 1.12;
      else if(bucket == 1) vol *= 0.95;
      else if(bucket == 2) vol *= 1.04;
      return FXAI_Clamp(vol, 0.70, 1.80);
   }

   double DynamicDriftBias(const double &f[]) const
   {
      int bucket = SessionBucket();
      double edge_bias = 0.08 * m_session_edge[bucket];
      double state_bias = 0.05 * m_edge_ema * FXAI_Sign(f[1]);
      return FXAI_ClipSym(edge_bias + state_bias, 0.35);
   }

   void BuildFeatureState(const double &x[], double &f[]) const
   {
      double a1 = SafeX(x, 1), a2 = SafeX(x, 2), a3 = SafeX(x, 3), a4 = SafeX(x, 4);
      double a5 = SafeX(x, 5), a6 = SafeX(x, 6), a7 = SafeX(x, 7), a8 = SafeX(x, 8);
      double a9 = SafeX(x, 9), a10 = SafeX(x, 10), a11 = SafeX(x, 11), a12 = SafeX(x, 12);

      double directional_bias   = FXAI_ClipSym(0.55*a1 + 0.30*a2 + 0.15*a3 - 0.10*a5, 6.0);
      double vol_pressure       = FXAI_ClipSym(0.75*MathAbs(a6) + 0.45*MathAbs(a7) + 0.12*MathAbs(a8), 6.0);
      double reversion_pressure = FXAI_ClipSym(-0.40*a1 + 0.36*a5 - 0.15*a10, 6.0);
      double breakout_hazard    = FXAI_ClipSym(0.45*MathAbs(a2 - a5) + 0.35*MathAbs(a3 - a4) + 0.10*MathAbs(a8), 6.0);
      double liquidity_noise    = FXAI_ClipSym(0.85*MathAbs(a7) + 0.25*MathAbs(a8) + 0.10*MathAbs(a11), 6.0);
      double carry_session      = 0.0;
      MqlDateTime dt;
      TimeToStruct(ResolveContextTime(), dt);
      if(dt.hour >= 6 && dt.hour < 12) carry_session = 0.12;
      else if(dt.hour >= 12 && dt.hour < 17) carry_session = 0.20;
      else if(dt.hour >= 17 && dt.hour < 21) carry_session = 0.08;
      else carry_session = -0.06;
      double asymmetry          = FXAI_ClipSym(0.30*a9 + 0.25*a10 - 0.18*a11 + 0.15*a12, 6.0);
      double friction_proxy     = FXAI_ClipSym(MathMax(0.0, ResolveCostPoints(x)), 6.0);

      f[0] = 1.0;
      f[1] = directional_bias;
      f[2] = vol_pressure;
      f[3] = reversion_pressure;
      f[4] = breakout_hazard;
      f[5] = liquidity_noise;
      f[6] = asymmetry + carry_session;
      f[7] = friction_proxy;
   }

   double NextUniform(uint &seed) const
   {
      seed = (uint)((ulong)1664525 * (ulong)seed + (ulong)1013904223);
      double u = (double)(seed % 16777216) / 16777215.0;
      return FXAI_Clamp(u, 1e-6, 1.0 - 1e-6);
   }

   double NextNoise(uint &seed) const
   {
      double u1 = NextUniform(seed);
      double u2 = NextUniform(seed);
      double r = MathSqrt(-2.0 * MathLog(u1));
      double th = 6.283185307179586 * u2;
      return FXAI_ClipSym(r * MathCos(th), 3.0);
   }

   void InferStateFromFeatures(const double &f[], double &out_state[]) const
   {
      for(int s=0; s<FXAI_CFXW_STATE; s++)
      {
         double z = m_trans[s] * m_state[s];
         for(int k=0; k<FXAI_CFXW_FEATS; k++)
            z += m_state_w[s][k] * f[k];
         out_state[s] = FXAI_Tanh(FXAI_ClipSym(z, 12.0));
      }
      // Explicit positivity for vol/noise/confidence channels.
      out_state[2] = FXAI_Clamp(0.5 + 0.5 * out_state[2], 0.0, 1.0);
      out_state[3] = FXAI_Clamp(0.5 + 0.5 * out_state[3], 0.0, 1.0);
      out_state[4] = FXAI_Clamp(0.5 + 0.5 * out_state[4], 0.0, 1.0);
      out_state[5] = FXAI_Clamp(0.5 + 0.5 * out_state[5], 0.0, 1.0);
   }

   void AdvanceParticle(double &st[], const double &f[], uint &seed, const double noise_scale, const double drift_bias, double &step_move) const
   {
      double eps1 = noise_scale * NextNoise(seed);
      double eps2 = noise_scale * NextNoise(seed);
      double eps3 = noise_scale * NextNoise(seed);

      double drift      = st[0];
      double reversion  = st[1];
      double vol        = FXAI_Clamp(st[2], 0.0, 2.0);
      double breakout   = FXAI_Clamp(st[3], 0.0, 2.0);
      double noise      = FXAI_Clamp(st[4], 0.0, 2.0);
      double confidence = FXAI_Clamp(st[5], 0.0, 2.0);

      double impulse = 0.60 * f[1] - 0.35 * f[3] + 0.20 * f[6] + drift_bias;
      double drift_next = 0.72 * drift + 0.18 * impulse + 0.06 * eps1 * (1.0 + breakout);
      double rev_next   = 0.62 * reversion + 0.14 * f[3] - 0.08 * drift + 0.05 * eps2;
      double vol_next   = FXAI_Clamp(0.82 * vol + 0.12 * MathAbs(f[2]) + 0.08 * MathAbs(eps1), 0.0, 2.5);
      double br_next    = FXAI_Clamp(0.64 * breakout + 0.16 * MathAbs(f[4]) + 0.06 * MathMax(0.0, MathAbs(impulse) - 0.5), 0.0, 2.5);
      double nz_next    = FXAI_Clamp(0.80 * noise + 0.12 * MathAbs(f[5]) + 0.06 * MathAbs(eps3), 0.0, 2.5);
      double cf_next    = FXAI_Clamp(0.88 * confidence + 0.08 * (1.0 - MathMin(1.0, nz_next / 2.5)), 0.0, 2.5);

      double directional = 0.90 * drift_next - 0.55 * rev_next + 0.35 * br_next * FXAI_Sign(impulse);
      double sigma = 0.12 + 0.30 * vol_next + 0.15 * nz_next;
      step_move = directional + sigma * eps2;

      st[0] = FXAI_ClipSym(drift_next, 2.5);
      st[1] = FXAI_ClipSym(rev_next,   2.5);
      st[2] = vol_next;
      st[3] = br_next;
      st[4] = nz_next;
      st[5] = cf_next;
   }

   void RolloutConsensus(const double &f[], const double cost_points, const double min_move_points, double &buy_mass, double &sell_mass, double &skip_mass, double &exp_move, double &adverse_mean) const
   {
      double cost = (cost_points >= 0.0 ? cost_points : 0.0);
      double mm = (min_move_points > 0.0 ? min_move_points : MathMax(0.10, cost));
      int particle_count = DynamicParticleCount(f, cost, mm);
      int horizon_steps = DynamicHorizonSteps(f);
      double noise_scale = DynamicNoiseScale(f);
      double drift_bias = DynamicDriftBias(f);

      double base_state[FXAI_CFXW_STATE];
      InferStateFromFeatures(f, base_state);

      int buy_n = 0, sell_n = 0, skip_n = 0;
      double edge_sum = 0.0;
      double adverse_pen_sum = 0.0;

      uint seed0 = (uint)(m_seed ^ (uint)ResolveContextTime());
      for(int p=0; p<particle_count; p++)
      {
         uint seed = (uint)(seed0 + (uint)(7331 * (p + 1)));
         double st[FXAI_CFXW_STATE];
         for(int s=0; s<FXAI_CFXW_STATE; s++) st[s] = base_state[s];

         double net = 0.0;
         double min_net = 0.0;
         double max_net = 0.0;
         for(int h=0; h<horizon_steps; h++)
         {
            double step = 0.0;
            AdvanceParticle(st, f, seed, noise_scale, drift_bias, step);
            net += step;
            if(net < min_net) min_net = net;
            if(net > max_net) max_net = net;
         }

         double favorable = MathMax(max_net, -min_net);
         double adverse = MathMax(MathAbs(min_net), MathAbs(max_net - net));
         double tradable = MathAbs(net) - cost;

         if(tradable <= 0.0 || favorable < mm)
         {
            skip_n++;
            adverse_pen_sum += MathAbs(adverse);
            continue;
         }

         if(net > 0.0) buy_n++;
         else if(net < 0.0) sell_n++;
         else skip_n++;

         edge_sum += tradable;
         adverse_pen_sum += adverse;
      }

      buy_mass = (double)buy_n  / (double)particle_count;
      sell_mass= (double)sell_n / (double)particle_count;
      skip_mass= (double)skip_n / (double)particle_count;

      double raw_edge = edge_sum / (double)particle_count;
      double risk_pen = adverse_pen_sum / (double)particle_count;
      adverse_mean = risk_pen;
      exp_move = MathMax(0.0, raw_edge - 0.35 * risk_pen);
   }

public:
   CFXAIAICFXWorld(void) : CFXAIAIPlugin()
   {
      Reset();
   }

   virtual int AIId(void) const { return AI_CFX_WORLD; }
   virtual string AIName(void) const { return "wm_cfx"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_WORLD_MODEL, caps, 8, 128);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      ResetModel();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      EnsureBootstrapped();
   }


   virtual bool PredictModelCore(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      EnsureBootstrapped();

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[FXAI_CFXW_FEATS];
      BuildFeatureState(xa, f);

      double cost = ResolveCostPoints(xa);
      if(cost < 0.0) cost = 0.0;
      double mm = ResolveMinMovePoints();
      if(mm <= 0.0) mm = MathMax(0.10, cost);
      double buy_mass = 0.0, sell_mass = 0.0, skip_mass = 0.0, exp_move = 0.0, adverse_mean = 0.0;
      RolloutConsensus(f, cost, mm, buy_mass, sell_mass, skip_mass, exp_move, adverse_mean);

      double total_dir = buy_mass + sell_mass;
      double consensus = MathAbs(buy_mass - sell_mass);
      double act = FXAI_Clamp(consensus + 0.35 * m_confidence_ema + 0.25 * m_consensus_ema, 0.0, 1.2);
      double edge_ratio = (exp_move - cost) / MathMax(mm, 0.10);
      double downside_ratio = adverse_mean / MathMax(exp_move + mm, 0.10);
      double risk_gate = 1.0 - FXAI_Clamp(0.55 * downside_ratio + 0.20 * m_downside_ema, 0.0, 0.90);
      double active = FXAI_Clamp((0.50 * FXAI_Sigmoid(1.25 * edge_ratio) + 0.35 * act + 0.15 * (1.0 - skip_mass)) * risk_gate, 0.0, 1.0);

      double p_buy = 0.5, p_sell = 0.5;
      if(total_dir > 1e-9)
      {
         p_buy = buy_mass / total_dir;
         p_sell = sell_mass / total_dir;
      }

      double p_raw3[3];
      p_raw3[(int)FXAI_LABEL_BUY]  = ClampProb(active * p_buy);
      p_raw3[(int)FXAI_LABEL_SELL] = ClampProb(active * p_sell);
      p_raw3[(int)FXAI_LABEL_SKIP] = ClampProb(MathMax(skip_mass, 1.0 - active));
      m_cal3.Calibrate(p_raw3, class_probs);
      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int c=0; c<3; c++) class_probs[c] /= s;

      expected_move_points = MathMax(0.0, exp_move - 0.20 * adverse_mean);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double probs[3];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return false;
      for(int c=0; c<3; c++)
         out.class_probs[c] = probs[c];

      double sigma = MathMax(0.10, 0.25 * ev + 0.55 * m_downside_ema + 0.20 * MathAbs(m_session_edge[SessionBucket()]));
      out.move_mean_points = MathMax(ev, (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.60 * sigma);
      out.move_q50_points = out.move_mean_points;
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.60 * sigma);
      out.confidence = FXAI_Clamp(0.45 * MathMax(probs[(int)FXAI_LABEL_BUY], probs[(int)FXAI_LABEL_SELL]) + 0.25 * m_confidence_ema + 0.20 * m_consensus_ema - 0.10 * FXAI_Clamp(m_downside_ema / MathMax(out.move_mean_points + 0.10, 0.10), 0.0, 1.0), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.50 + 0.20 * m_consensus_ema + 0.15 * m_confidence_ema + 0.15 * FXAI_Clamp(m_edge_ema / MathMax(out.move_mean_points + 0.10, 0.10), 0.0, 1.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      return true;
   }

protected:
   virtual void TrainModelCore(const int y, const double &x[], const FXAIAIHyperParams &hp, const double move_points)
   {
      EnsureBootstrapped();

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[FXAI_CFXW_FEATS];
      BuildFeatureState(xa, f);

      double inferred[FXAI_CFXW_STATE];
      InferStateFromFeatures(f, inferred);

      double cost = ResolveCostPoints(xa);
      if(cost < 0.0) cost = 0.0;
      double edge = MathMax(0.0, MathAbs(move_points) - cost);
      int dir = 0;
      if(y == (int)FXAI_LABEL_BUY) dir = 1;
      else if(y == (int)FXAI_LABEL_SELL) dir = -1;
      else if(move_points > 0.0) dir = 1;
      else if(move_points < 0.0) dir = -1;

      double target[FXAI_CFXW_STATE];
      target[0] = FXAI_ClipSym(move_points / MathMax(cost + 1.0, 1.0), 2.0);
      target[1] = FXAI_ClipSym(-0.65 * target[0] + 0.15 * f[3], 2.0);
      target[2] = FXAI_Clamp(0.20 + 0.30 * MathAbs(f[2]) + 0.15 * edge, 0.0, 1.5);
      target[3] = FXAI_Clamp(0.10 + 0.25 * MathAbs(f[4]) + 0.10 * edge, 0.0, 1.5);
      target[4] = FXAI_Clamp(0.10 + 0.25 * MathAbs(f[5]) + 0.10 * cost, 0.0, 1.5);
      target[5] = FXAI_Clamp((dir == 0 ? 0.45 : 0.65) + (edge > 0.0 ? 0.15 : -0.10), 0.0, 1.5);

      double buy_mass = 0.0, sell_mass = 0.0, skip_mass = 0.0, exp_move = 0.0, adverse_mean = 0.0;
      RolloutConsensus(f, cost, MathMax(0.10, cost), buy_mass, sell_mass, skip_mass, exp_move, adverse_mean);
      double downside_pen = adverse_mean / MathMax(MathAbs(move_points) + cost + 0.10, 0.10);
      target[3] = FXAI_Clamp(target[3] + 0.12 * downside_pen, 0.0, 1.8);
      target[4] = FXAI_Clamp(target[4] + 0.10 * downside_pen, 0.0, 1.8);
      target[5] = FXAI_Clamp(target[5] - 0.15 * downside_pen, 0.0, 1.5);

      double sw = FXAI_Clamp(MoveSampleWeight(x, move_points), 0.25, 4.0);
      double lr = FXAI_Clamp(0.20 * hp.lr * sw, 0.0002, 0.04);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.05);
      double total_dir = buy_mass + sell_mass;
      double consensus = MathAbs(buy_mass - sell_mass);
      double act = FXAI_Clamp(consensus + 0.35 * m_confidence_ema + 0.25 * m_consensus_ema, 0.0, 1.2);
      double mm = ResolveMinMovePoints();
      if(mm <= 0.0) mm = MathMax(0.10, cost);
      double edge_ratio = (exp_move - cost) / MathMax(mm, 0.10);
      double downside_ratio = adverse_mean / MathMax(exp_move + mm, 0.10);
      double risk_gate = 1.0 - FXAI_Clamp(0.55 * downside_ratio + 0.20 * m_downside_ema, 0.0, 0.90);
      double active = FXAI_Clamp((0.50 * FXAI_Sigmoid(1.25 * edge_ratio) + 0.35 * act + 0.15 * (1.0 - skip_mass)) * risk_gate, 0.0, 1.0);
      double p_buy = 0.5, p_sell = 0.5;
      if(total_dir > 1e-9)
      {
         p_buy = buy_mass / total_dir;
         p_sell = sell_mass / total_dir;
      }
      double p_raw3[3];
      p_raw3[(int)FXAI_LABEL_BUY] = ClampProb(active * p_buy);
      p_raw3[(int)FXAI_LABEL_SELL] = ClampProb(active * p_sell);
      p_raw3[(int)FXAI_LABEL_SKIP] = ClampProb(MathMax(skip_mass, 1.0 - active));
      NormalizeClassDistribution(p_raw3);
      int cls = (dir == 0 ? (int)FXAI_LABEL_SKIP : (dir > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL));
      m_cal3.Update(p_raw3, cls, sw, lr);

      for(int s=0; s<FXAI_CFXW_STATE; s++)
      {
         double err = target[s] - inferred[s];
         for(int k=0; k<FXAI_CFXW_FEATS; k++)
         {
            double grad = err * f[k] - l2 * m_state_w[s][k];
            m_state_g2[s][k] += grad * grad;
            double step = lr / MathSqrt(1.0 + m_state_g2[s][k]);
            m_state_w[s][k] = FXAI_ClipSym(m_state_w[s][k] + step * grad, 4.0);
         }
         double new_state = 0.92 * m_state[s] + 0.08 * target[s];
         if(s >= 2) new_state = FXAI_Clamp(new_state, 0.0, 2.0);
         else new_state = FXAI_ClipSym(new_state, 2.5);
         m_state[s] = new_state;
      }

      double pred_sign = 0.0;
      if(buy_mass > sell_mass) pred_sign = 1.0;
      else if(sell_mass > buy_mass) pred_sign = -1.0;

      double hit = 0.5;
      if(dir != 0) hit = (pred_sign == (double)dir ? 1.0 : 0.0);
      m_consensus_ema  = 0.97 * m_consensus_ema  + 0.03 * MathAbs(buy_mass - sell_mass);
      m_confidence_ema = 0.97 * m_confidence_ema + 0.03 * (1.0 - skip_mass);
      m_edge_ema       = 0.96 * m_edge_ema       + 0.04 * edge;
      m_downside_ema   = 0.96 * m_downside_ema   + 0.04 * adverse_mean;
      FXAI_UpdateMoveEMA(m_edge_ema, m_move_ready, edge, 0.02);

      int sb = SessionBucket();
      double signed_edge = (dir == 0 ? -0.20 * adverse_mean : (double)dir * edge);
      m_session_edge[sb] = 0.97 * m_session_edge[sb] + 0.03 * signed_edge;

      // Transition parameter adaptation.
      double sess_adj = 0.02 * MathAbs(m_session_edge[sb]);
      m_trans[0] = FXAI_Clamp(0.995 * m_trans[0] + 0.005 * (0.55 + 0.20 * MathAbs(target[0]) + sess_adj), 0.20, 0.98);
      m_trans[1] = FXAI_Clamp(0.995 * m_trans[1] + 0.005 * (0.45 + 0.15 * MathAbs(target[1])), 0.10, 0.95);
      m_trans[2] = FXAI_Clamp(0.995 * m_trans[2] + 0.005 * (0.70 + 0.10 * target[2] + 0.08 * downside_pen), 0.20, 0.995);
      m_trans[3] = FXAI_Clamp(0.995 * m_trans[3] + 0.005 * (0.40 + 0.18 * target[3] + 0.05 * downside_pen), 0.10, 0.95);
      m_trans[4] = FXAI_Clamp(0.995 * m_trans[4] + 0.005 * (0.65 + 0.10 * target[4] + 0.06 * downside_pen), 0.20, 0.995);
      m_trans[5] = FXAI_Clamp(0.995 * m_trans[5] + 0.005 * (0.78 + 0.08 * target[5] - 0.04 * downside_pen), 0.20, 0.999);

      m_seed = (uint)((ulong)1103515245 * (ulong)(m_seed + (uint)(dir + 3)) + (ulong)12345);
      m_steps++;
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3];
      double exp_move = 0.0;
      if(PredictModelCore(x, hp, probs, exp_move))
         return probs[(int)FXAI_LABEL_BUY] / MathMax(probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL], 1e-6);
      return 0.50;
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureBootstrapped();
      FXAIAIModelOutputV4 out;
      if(PredictDistributionCore(x, hp, out) && out.move_mean_points > 0.0)
         return out.move_mean_points;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return 0.0;
   }
};

#endif // __FXAI_AI_CFX_WORLD_MQH__
