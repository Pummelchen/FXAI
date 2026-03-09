// FXAI v2
#ifndef __FXAI_AI_M1SYNC_MQH__
#define __FXAI_AI_M1SYNC_MQH__

#include "..\plugin_base.mqh"

class CFXAIAIM1Sync : public CFXAIAIPlugin
{
private:
   bool   m_init;
   int    m_steps;
   double m_hit_ema;
   double m_edge_ema;

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

   int EvaluateSyncSignal(const datetime ctx_time,
                          double &expected_move_points,
                          double &strength) const
   {
      expected_move_points = 0.0;
      strength = 0.0;

      string symbol = _Symbol;
      if(!SymbolSelect(symbol, true)) return (int)FXAI_LABEL_SKIP;

      int shift = -1;
      if(ctx_time > 0) shift = iBarShift(symbol, PERIOD_M1, ctx_time, true);
      if(shift < 1 && ctx_time > 0) shift = iBarShift(symbol, PERIOD_M1, ctx_time, false);
      if(shift < 1) shift = 1;

      double c1 = iClose(symbol, PERIOD_M1, shift);
      double c2 = iClose(symbol, PERIOD_M1, shift + 1);
      double c3 = iClose(symbol, PERIOD_M1, shift + 2);
      if(c1 <= 0.0 || c2 <= 0.0 || c3 <= 0.0)
         return (int)FXAI_LABEL_SKIP;

      double now_price = ResolveCurrentLikePrice(symbol, shift, ctx_time);
      if(now_price <= 0.0)
         return (int)FXAI_LABEL_SKIP;

      double point = ResolvePointSize(symbol);
      double cost_points = MathMax(0.0, m_ctx_cost_points);
      double mm = ResolveMinMovePoints();
      if(mm <= 0.0) mm = MathMax(0.10, cost_points);
      double eps = MathMax(0.10 * point, 0.02 * cost_points * point);

      double s1 = c2 - c3;
      double s2 = c1 - c2;
      double s3 = now_price - c1;

      bool up_chain = (s1 > eps && s2 > eps && s3 > eps);
      bool down_chain = (s1 < -eps && s2 < -eps && s3 < -eps);
      if(!up_chain && !down_chain)
         return (int)FXAI_LABEL_SKIP;

      double step1_points = MathAbs(s1) / point;
      double step2_points = MathAbs(s2) / point;
      double step3_points = MathAbs(s3) / point;
      double total_points = MathAbs(now_price - c3) / point;
      double min_step_points = MathMin(step1_points, MathMin(step2_points, step3_points));

      expected_move_points = MathMax(0.0, total_points - cost_points);
      double total_score = FXAI_Sigmoid((expected_move_points / MathMax(mm, 0.10)) - 0.50);
      double step_score = FXAI_Sigmoid((min_step_points / MathMax(mm, 0.10)) - 0.25);
      strength = FXAI_Clamp(0.55 * total_score + 0.45 * step_score, 0.0, 1.0);

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
   virtual string AIName(void) const { return "m1sync"; }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_init = false;
      m_steps = 0;
      m_hit_ema = 0.55;
      m_edge_ema = 0.0;
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      m_init = true;
   }

   virtual bool SupportsNativeClassProbs(void) const
   {
      return true;
   }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double strength = 0.0;
      int signal = EvaluateSyncSignal(ResolveContextTime(), expected_move_points, strength);
      double reliability = FXAI_Clamp(m_hit_ema, 0.25, 0.95);

      if(signal == (int)FXAI_LABEL_BUY)
      {
         double buy = FXAI_Clamp(0.78 + 0.14 * strength + 0.08 * (reliability - 0.50), 0.65, 0.97);
         class_probs[(int)FXAI_LABEL_BUY] = buy;
         class_probs[(int)FXAI_LABEL_SELL] = 0.01;
         class_probs[(int)FXAI_LABEL_SKIP] = MathMax(0.02, 1.0 - buy - 0.01);
      }
      else if(signal == (int)FXAI_LABEL_SELL)
      {
         double sell = FXAI_Clamp(0.78 + 0.14 * strength + 0.08 * (reliability - 0.50), 0.65, 0.97);
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
   virtual void UpdateWithMove(const int y,
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
      PredictNativeClassProbs(x, hp, probs, exp_move);
      return probs[(int)FXAI_LABEL_BUY] / MathMax(probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL], 1e-6);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3];
      double exp_move = 0.0;
      PredictNativeClassProbs(x, hp, probs, exp_move);
      return exp_move;
   }
};

#endif // __FXAI_AI_M1SYNC_MQH__
