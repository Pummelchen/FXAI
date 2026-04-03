int FXAI_CoreClampHorizon(const int horizon_minutes)
{
   int h = horizon_minutes;
   if(h < 1) h = 1;
   if(h > 1440) h = 1440;
   return h;
}

double FXAI_SymbolModelScale(const string symbol)
{
   string s = symbol;
   StringToUpper(s);
   if(StringFind(s, "XAU") >= 0 || StringFind(s, "GOLD") >= 0 ||
      StringFind(s, "XAG") >= 0 || StringFind(s, "SILVER") >= 0)
      return 1.18;
   if(StringFind(s, "US30") >= 0 || StringFind(s, "NAS") >= 0 ||
      StringFind(s, "SPX") >= 0 || StringFind(s, "DAX") >= 0 ||
      StringFind(s, "GER40") >= 0 || StringFind(s, "JP225") >= 0)
      return 1.15;
   if(StringFind(s, "OIL") >= 0 || StringFind(s, "WTI") >= 0 ||
      StringFind(s, "BRENT") >= 0 || StringFind(s, "NGAS") >= 0)
      return 1.12;
   if(StringFind(s, "JPY") >= 0 || StringFind(s, "GBP") >= 0)
      return 1.06;
   return 1.00;
}

double FXAI_HorizonModelScale(const int horizon_minutes)
{
   int h = FXAI_CoreClampHorizon(horizon_minutes);
   if(h <= 5) return 0.92;
   if(h <= 15) return 0.98;
   if(h <= 60) return 1.00;
   if(h <= 240) return 1.10;
   return 1.18;
}

double FXAI_ModelCapacityScale(const string symbol,
                               const int horizon_minutes)
{
   return FXAI_Clamp(FXAI_SymbolModelScale(symbol) * FXAI_HorizonModelScale(horizon_minutes), 0.85, 1.35);
}

int FXAI_ContextSequenceSpan(const int max_cap,
                             const int horizon_minutes,
                             const string symbol,
                             const int base_min = 8)
{
   int cap = MathMax(base_min, max_cap);
   double scale = FXAI_ModelCapacityScale(symbol, horizon_minutes);
   int span = (int)MathRound((double)cap * FXAI_Clamp(0.55 + 0.35 * scale, 0.45, 1.10));
   if(span < base_min) span = base_min;
   if(span > max_cap) span = max_cap;
   return span;
}

int FXAI_ContextBatchSpan(const int max_cap,
                          const int horizon_minutes,
                          const string symbol,
                          const int base_min = 4)
{
   int cap = MathMax(base_min, max_cap);
   double scale = FXAI_ModelCapacityScale(symbol, horizon_minutes);
   int span = (int)MathRound((double)cap * FXAI_Clamp(0.60 + 0.30 * scale, 0.45, 1.00));
   if(span < base_min) span = base_min;
   if(span > max_cap) span = max_cap;
   return span;
}

int FXAI_ContextTreeBudget(const int max_cap,
                           const int horizon_minutes,
                           const string symbol,
                           const int base_min)
{
   int cap = MathMax(base_min, max_cap);
   double scale = FXAI_ModelCapacityScale(symbol, horizon_minutes);
   int budget = (int)MathRound((double)cap * FXAI_Clamp(0.55 + 0.40 * scale, 0.50, 1.15));
   if(budget < base_min) budget = base_min;
   if(budget > max_cap) budget = max_cap;
   return budget;
}

void FXAI_UpdateMoveEMA(double &ema_abs_move,
                       bool &ready,
                       const double move_points,
                       const double alpha)
{
   double a = FXAI_Clamp(alpha, 0.001, 0.500);
   double v = MathAbs(move_points);
   if(!MathIsValidNumber(v)) return;

   if(!ready)
   {
      ema_abs_move = v;
      ready = true;
      return;
   }

   ema_abs_move = (1.0 - a) * ema_abs_move + a * v;
}

int FXAI_ThreeWayBranch(const double x, const double split)
{
   if(x < split - 0.50) return 0;
   if(x > split + 0.50) return 2;
   return 1;
}

