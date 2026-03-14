#ifndef __FXAI_AI_M1SYNC_MQH__
#define __FXAI_AI_M1SYNC_MQH__

#include "..\API\plugin_base.mqh"

class CFXAIAIM1Sync : public CFXAIAIPlugin
{
private:
   bool   m_init;
   int    m_steps;
   double m_hit_ema;
   double m_edge_ema;
   bool   m_has_synth;
   datetime m_synth_time[];
   double m_synth_open[];
   double m_synth_high[];
   double m_synth_low[];
   double m_synth_close[];

   double ResolvePointSize(const string symbol) const
   {
      double pt = SymbolInfoDouble(symbol, SYMBOL_POINT);
      if(pt <= 0.0) pt = _Point;
      if(pt <= 0.0) pt = 0.00001;
      return pt;
   }

   double ResolveCurrentLikePrice(const string symbol,
                                  const int shift,
                                  const datetime ctx_time) const
   {
      if(m_has_synth)
      {
         if(shift > 0 && shift - 1 < ArraySize(m_synth_open))
         {
            double op_s = m_synth_open[shift - 1];
            if(op_s > 0.0) return op_s;
         }
         if(shift >= 0 && shift < ArraySize(m_synth_close))
         {
            double cl_s = m_synth_close[shift];
            if(cl_s > 0.0) return cl_s;
         }
         return 0.0;
      }

      datetime latest_closed = iTime(symbol, PERIOD_M1, 1);
      bool use_live = (ctx_time <= 0 || latest_closed <= 0 || MathAbs((double)(latest_closed - ctx_time)) <= 1.0);
      if(use_live)
      {
         MqlTick tick;
         if(SymbolInfoTick(symbol, tick))
         {
            if(tick.bid > 0.0 && tick.ask > 0.0) return 0.5 * (tick.bid + tick.ask);
            if(tick.last > 0.0) return tick.last;
         }
      }

      if(shift > 0)
      {
         double op = iOpen(symbol, PERIOD_M1, shift - 1);
         if(op > 0.0) return op;
      }

      double fallback = iClose(symbol, PERIOD_M1, MathMax(shift - 1, 0));
      if(fallback > 0.0) return fallback;
      return 0.0;
   }

   int FindSyntheticShift(const datetime ctx_time) const
   {
      int n = ArraySize(m_synth_time);
      if(n <= 0) return -1;
      if(ctx_time <= 0) return 1;

      for(int i=0; i<n; i++)
      {
         datetime t = m_synth_time[i];
         if(t <= 0) continue;
         if(t <= ctx_time) return i;
      }
      return n - 1;
   }

   int EvaluateSyncSignal(const datetime ctx_time,
                          double &expected_move_points,
                          double &strength) const
   {
      expected_move_points = 0.0;
      strength = 0.0;

      string symbol = _Symbol;
      if(!m_has_synth && !SymbolSelect(symbol, true)) return (int)FXAI_LABEL_SKIP;

      int shift = -1;
      if(m_has_synth)
         shift = FindSyntheticShift(ctx_time);
      else if(ctx_time > 0)
         shift = iBarShift(symbol, PERIOD_M1, ctx_time, true);
      if(!m_has_synth && shift < 1 && ctx_time > 0) shift = iBarShift(symbol, PERIOD_M1, ctx_time, false);
      if(shift < 1) shift = 1;
      int bars = FXAI_GetM1SyncBars();

      double closes[];
      ArrayResize(closes, bars);
      for(int i=0; i<bars; i++)
      {
         int bar_shift = shift + (bars - 1 - i);
         if(m_has_synth)
         {
            if(bar_shift < 0 || bar_shift >= ArraySize(m_synth_close))
               return (int)FXAI_LABEL_SKIP;
            closes[i] = m_synth_close[bar_shift];
         }
         else
         {
            closes[i] = iClose(symbol, PERIOD_M1, bar_shift);
         }
         if(closes[i] <= 0.0)
            return (int)FXAI_LABEL_SKIP;
      }

      double now_price = ResolveCurrentLikePrice(symbol, shift, ctx_time);
      if(now_price <= 0.0)
         return (int)FXAI_LABEL_SKIP;

      double point = ResolvePointSize(symbol);
      double cost_points = MathMax(0.0, m_ctx_cost_points);
      double mm = ResolveMinMovePoints();
      if(mm <= 0.0) mm = MathMax(0.10, cost_points);
      double eps = MathMax(0.10 * point, 0.02 * cost_points * point);

      bool up_chain = true;
      bool down_chain = true;
      double min_step_points = DBL_MAX;
      double prev = closes[0];
      for(int i=1; i<bars; i++)
      {
         double step = closes[i] - prev;
         if(step <= eps) up_chain = false;
         if(step >= -eps) down_chain = false;
         double step_points = MathAbs(step) / point;
         if(step_points < min_step_points) min_step_points = step_points;
         prev = closes[i];
      }

      double final_step = now_price - closes[bars - 1];
      if(final_step <= eps) up_chain = false;
      if(final_step >= -eps) down_chain = false;
      double final_step_points = MathAbs(final_step) / point;
      if(final_step_points < min_step_points) min_step_points = final_step_points;

      if(!up_chain && !down_chain)
         return (int)FXAI_LABEL_SKIP;

      double total_points = MathAbs(now_price - closes[0]) / point;
      if(min_step_points == DBL_MAX) min_step_points = final_step_points;

      // The shared framework EV gate already subtracts trade cost. This rule plugin
      // should therefore publish the raw move amplitude once, not net-of-cost twice.
      expected_move_points = MathMax(total_points, 0.0);
      double edge_points = total_points - cost_points;
      double total_score = FXAI_Sigmoid(edge_points / MathMax(mm, 0.10));
      double step_score = FXAI_Sigmoid((min_step_points / MathMax(mm, 0.10)) - 0.15);
      strength = FXAI_Clamp(0.60 * total_score + 0.40 * step_score, 0.0, 1.0);

      if(up_chain) return (int)FXAI_LABEL_BUY;
      if(down_chain) return (int)FXAI_LABEL_SELL;
      return (int)FXAI_LABEL_SKIP;
   }

   int NormalizeObservedLabel(const int y, const double move_points) const
   {
      if(y == (int)FXAI_LABEL_BUY || y == (int)FXAI_LABEL_SELL || y == (int)FXAI_LABEL_SKIP)
         return y;
      if(move_points > 0.0) return (int)FXAI_LABEL_BUY;
      if(move_points < 0.0) return (int)FXAI_LABEL_SELL;
      return (int)FXAI_LABEL_SKIP;
   }

public:
   CFXAIAIM1Sync(void) : CFXAIAIPlugin()
   {
      Reset();
   }

   virtual int AIId(void) const { return (int)AI_M1SYNC; }
   virtual string AIName(void) const { return "rule_m1sync"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_RULE_BASED, caps, 1, 1);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_init = false;
      m_steps = 0;
      m_hit_ema = 0.55;
      m_edge_ema = 0.0;
      m_has_synth = false;
      ArrayResize(m_synth_time, 0);
      ArrayResize(m_synth_open, 0);
      ArrayResize(m_synth_high, 0);
      ArrayResize(m_synth_low, 0);
      ArrayResize(m_synth_close, 0);
   }

   virtual bool SupportsSyntheticSeries(void) const { return true; }

   virtual bool SetSyntheticSeries(const datetime &time_arr[],
                                   const double &open_arr[],
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[])
   {
      int n = ArraySize(time_arr);
      if(n <= 0 || ArraySize(open_arr) != n || ArraySize(high_arr) != n ||
         ArraySize(low_arr) != n || ArraySize(close_arr) != n)
         return false;

      ArrayCopy(m_synth_time, time_arr);
      ArrayCopy(m_synth_open, open_arr);
      ArrayCopy(m_synth_high, high_arr);
      ArrayCopy(m_synth_low, low_arr);
      ArrayCopy(m_synth_close, close_arr);
      ArraySetAsSeries(m_synth_time, true);
      ArraySetAsSeries(m_synth_open, true);
      ArraySetAsSeries(m_synth_high, true);
      ArraySetAsSeries(m_synth_low, true);
      ArraySetAsSeries(m_synth_close, true);
      m_has_synth = true;
      return true;
   }

   virtual void ClearSyntheticSeries(void)
   {
      m_has_synth = false;
      ArrayResize(m_synth_time, 0);
      ArrayResize(m_synth_open, 0);
      ArrayResize(m_synth_high, 0);
      ArrayResize(m_synth_low, 0);
      ArrayResize(m_synth_close, 0);
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      m_init = true;
   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double strength = 0.0;
      int signal = EvaluateSyncSignal(ResolveContextTime(), expected_move_points, strength);
      double reliability = FXAI_Clamp(m_hit_ema, 0.25, 0.95);
      double cost_points = ResolveCostPoints(x);
      double move_scale = MathMax(expected_move_points, MathMax(ResolveMinMovePoints(), 0.10));
      double execution_drag = FXAI_Clamp(cost_points / MathMax(move_scale, 0.25), 0.0, 1.5);
      int sess = ContextSessionBucket();
      double session_penalty = ((sess == 0 || sess == FXAI_PLUGIN_SESSION_BUCKETS - 1) ? 0.10 : 0.0);
      reliability = FXAI_Clamp(reliability * (1.0 - 0.22 * execution_drag - session_penalty), 0.10, 0.95);
      strength = FXAI_Clamp(strength * (1.0 - 0.28 * execution_drag - 0.50 * session_penalty), 0.0, 1.0);
      expected_move_points = MathMax(0.0, expected_move_points * (1.0 - 0.25 * execution_drag - 0.35 * session_penalty));
      if(execution_drag >= 0.95)
      {
         signal = (int)FXAI_LABEL_SKIP;
         expected_move_points = 0.0;
         strength = 0.0;
      }

      if(signal == (int)FXAI_LABEL_BUY)
      {
         double buy = FXAI_Clamp(0.90 + 0.08 * strength + 0.04 * (reliability - 0.50), 0.85, 0.995);
         class_probs[(int)FXAI_LABEL_BUY] = buy;
         class_probs[(int)FXAI_LABEL_SELL] = 0.01;
         class_probs[(int)FXAI_LABEL_SKIP] = MathMax(0.02, 1.0 - buy - 0.01);
      }
      else if(signal == (int)FXAI_LABEL_SELL)
      {
         double sell = FXAI_Clamp(0.90 + 0.08 * strength + 0.04 * (reliability - 0.50), 0.85, 0.995);
         class_probs[(int)FXAI_LABEL_SELL] = sell;
         class_probs[(int)FXAI_LABEL_BUY] = 0.01;
         class_probs[(int)FXAI_LABEL_SKIP] = MathMax(0.02, 1.0 - sell - 0.01);
      }
      else
      {
         class_probs[(int)FXAI_LABEL_BUY] = 0.02;
         class_probs[(int)FXAI_LABEL_SELL] = 0.02;
         class_probs[(int)FXAI_LABEL_SKIP] = 0.96;
         expected_move_points = 0.0;
      }

      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int i=0; i<3; i++) class_probs[i] /= s;
      return true;
   }

protected:
   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      double expected_move_points = 0.0;
      double strength = 0.0;
      int signal = EvaluateSyncSignal(ResolveContextTime(), expected_move_points, strength);
      int observed = NormalizeObservedLabel(y, move_points);

      double hit = 0.5;
      if(signal == (int)FXAI_LABEL_SKIP)
         hit = (observed == (int)FXAI_LABEL_SKIP ? 1.0 : 0.0);
      else
         hit = (signal == observed ? 1.0 : 0.0);

      m_hit_ema = 0.985 * m_hit_ema + 0.015 * hit;
      m_edge_ema = 0.980 * m_edge_ema + 0.020 * expected_move_points;
      FXAI_UpdateMoveEMA(m_edge_ema, m_move_ready, expected_move_points, 0.02);
      m_steps++;
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3];
      double exp_move = 0.0;
      PredictModelCore(x, hp, probs, exp_move);
      return probs[(int)FXAI_LABEL_BUY] / MathMax(probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL], 1e-6);
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      if(!PredictModelCore(x, hp, out.class_probs, out.move_mean_points))
         return false;
      double sigma = MathMax(0.10, 0.35 * out.move_mean_points);
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      out.confidence = MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]);
      out.reliability = FXAI_Clamp(m_hit_ema, 0.25, 0.95);
      out.has_quantiles = true;
      out.has_confidence = true;
      PopulatePathQualityHeads(out, x, FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0), out.reliability, out.confidence);
      return true;
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3];
      double exp_move = 0.0;
      PredictModelCore(x, hp, probs, exp_move);
      return exp_move;
   }
};

#endif // __FXAI_AI_M1SYNC_MQH__
