int FXAI_ContextSymbolCategory(const string symbol)
{
   string sym = symbol;
   StringToUpper(sym);

   if(StringFind(sym, "XAU") >= 0 || StringFind(sym, "XAG") >= 0 ||
      StringFind(sym, "XPT") >= 0 || StringFind(sym, "XPD") >= 0)
      return FXAI_CONTEXT_CAT_METAL;

   if(StringFind(sym, "US30") >= 0 || StringFind(sym, "DE40") >= 0 ||
      StringFind(sym, "GER40") >= 0 || StringFind(sym, "JP225") >= 0 ||
      StringFind(sym, "NAS100") >= 0 || StringFind(sym, "USTEC") >= 0 ||
      StringFind(sym, "SPX500") >= 0 || StringFind(sym, "US500") >= 0 ||
      StringFind(sym, "UK100") >= 0 || StringFind(sym, "HK50") >= 0 ||
      StringFind(sym, "AUS200") >= 0 || StringFind(sym, "FRA40") >= 0)
      return FXAI_CONTEXT_CAT_INDEX;

   if(StringFind(sym, "WTI") >= 0 || StringFind(sym, "XTI") >= 0 ||
      StringFind(sym, "BRENT") >= 0 || StringFind(sym, "NATGAS") >= 0 ||
      StringFind(sym, "NGAS") >= 0)
      return FXAI_CONTEXT_CAT_ENERGY;

   if(StringFind(sym, "BTC") >= 0 || StringFind(sym, "ETH") >= 0 ||
      StringFind(sym, "XRP") >= 0 || StringFind(sym, "SOL") >= 0 ||
      StringFind(sym, "ADA") >= 0 || StringFind(sym, "LTC") >= 0)
      return FXAI_CONTEXT_CAT_CRYPTO;

   if(StringFind(sym, "VIX") >= 0 || StringFind(sym, "VOL") >= 0 ||
      StringFind(sym, "DXY") >= 0 || StringFind(sym, "USDX") >= 0 ||
      StringFind(sym, "DOLLAR") >= 0 || StringFind(sym, "TNX") >= 0 ||
      StringFind(sym, "USB10") >= 0 || StringFind(sym, "US10Y") >= 0 ||
      StringFind(sym, "US02Y") >= 0 || StringFind(sym, "BUND") >= 0 ||
      StringFind(sym, "GILT") >= 0 || StringFind(sym, "JGB") >= 0)
      return FXAI_CONTEXT_CAT_RISK;

   if(StringLen(sym) >= 6)
   {
      string a = StringSubstr(sym, 0, 3);
      string b = StringSubstr(sym, 3, 3);
      bool a_alpha = true;
      bool b_alpha = true;
      for(int i=0; i<3; i++)
      {
         ushort ca = StringGetCharacter(a, i);
         ushort cb = StringGetCharacter(b, i);
         if(ca < 'A' || ca > 'Z') a_alpha = false;
         if(cb < 'A' || cb > 'Z') b_alpha = false;
      }
      if(a_alpha && b_alpha)
         return FXAI_CONTEXT_CAT_FX;
   }

   return FXAI_CONTEXT_CAT_OTHER;
}

double FXAI_ContextCategoryPriority(const int category)
{
   switch(category)
   {
      case FXAI_CONTEXT_CAT_FX: return 1.00;
      case FXAI_CONTEXT_CAT_METAL: return 0.92;
      case FXAI_CONTEXT_CAT_INDEX: return 0.88;
      case FXAI_CONTEXT_CAT_ENERGY: return 0.76;
      case FXAI_CONTEXT_CAT_CRYPTO: return 0.60;
      case FXAI_CONTEXT_CAT_RISK: return 0.94;
      default: return 0.50;
   }
}

double FXAI_ContextSharedSymbolScore(const string main_symbol,
                                     const string candidate_symbol)
{
   string main_sym = main_symbol;
   string cand_sym = candidate_symbol;
   StringToUpper(main_sym);
   StringToUpper(cand_sym);

   if(StringLen(main_sym) < 6 || StringLen(cand_sym) < 6)
      return 0.0;

   string main_a = StringSubstr(main_sym, 0, 3);
   string main_b = StringSubstr(main_sym, 3, 3);
   string cand_a = StringSubstr(cand_sym, 0, 3);
   string cand_b = StringSubstr(cand_sym, 3, 3);

   double score = 0.0;
   if(main_a == cand_a || main_a == cand_b) score += 0.35;
   if(main_b == cand_a || main_b == cand_b) score += 0.35;
   if(StringFind(cand_sym, main_a) >= 0) score += 0.10;
   if(StringFind(cand_sym, main_b) >= 0) score += 0.10;
   return FXAI_Clamp(score, 0.0, 1.0);
}

double FXAI_ContextLiquidityScore(const string symbol)
{
   if(!SymbolSelect(symbol, true))
      return -1.0;

   long trade_mode = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
   if(trade_mode == SYMBOL_TRADE_MODE_DISABLED)
      return -1.0;

   MqlTick tick;
   if(!FXAI_MarketDataGetLatestTick(symbol, tick))
      return 0.20;

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0) point = 0.0001;
   double spread_pts = 0.0;
   if(tick.ask > 0.0 && tick.bid > 0.0)
      spread_pts = (tick.ask - tick.bid) / point;
   if(spread_pts <= 0.0)
      spread_pts = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD);

   return FXAI_Clamp(1.25 - 0.06 * spread_pts, 0.0, 1.0);
}

double FXAI_ContextDataHealthScore(const string symbol)
{
   if(!SymbolSelect(symbol, true))
      return -1.0;

   MqlRates rates[];
   if(!FXAI_MarketDataCopyRatesByPos(symbol, PERIOD_M1, 0, 4, rates) || ArraySize(rates) <= 0)
      return 0.10;

   datetime last_bar = rates[0].time;
   datetime now = TimeCurrent();
   if(now <= 0) now = TimeTradeServer();
   if(now <= 0) now = last_bar;
   double age_minutes = (double)MathAbs((int)(now - last_bar)) / 60.0;
   double freshness = FXAI_Clamp(1.15 - 0.02 * age_minutes, 0.0, 1.0);
   return freshness;
}

double FXAI_ContextSessionOverlapScore(const string main_symbol,
                                       const string candidate_symbol)
{
   MqlDateTime dt;
   datetime now = TimeCurrent();
   if(now <= 0) now = TimeTradeServer();
   TimeToStruct(now, dt);
   int session = FXAI_DeriveSessionBucket(now);
   int cat = FXAI_ContextSymbolCategory(candidate_symbol);
   string main_sym = main_symbol;
   string cand_sym = candidate_symbol;
   StringToUpper(main_sym);
   StringToUpper(cand_sym);

   double score = 0.45;
   if(cat == FXAI_CONTEXT_CAT_FX)
   {
      if(session == 0 && (StringFind(cand_sym, "JPY") >= 0 || StringFind(cand_sym, "AUD") >= 0 || StringFind(cand_sym, "NZD") >= 0))
         score += 0.35;
      if(session == 1 && (StringFind(cand_sym, "EUR") >= 0 || StringFind(cand_sym, "GBP") >= 0 || StringFind(cand_sym, "CHF") >= 0))
         score += 0.35;
      if(session == 2 && (StringFind(cand_sym, "USD") >= 0 || StringFind(cand_sym, "CAD") >= 0))
         score += 0.35;
   }
   else if(cat == FXAI_CONTEXT_CAT_METAL || cat == FXAI_CONTEXT_CAT_RISK)
   {
      if(session >= 1) score += 0.30;
   }
   else if(cat == FXAI_CONTEXT_CAT_INDEX || cat == FXAI_CONTEXT_CAT_ENERGY)
   {
      if(session == 2) score += 0.30;
      else if(session == 1) score += 0.15;
   }

   if(StringFind(cand_sym, StringSubstr(main_sym, 0, 3)) >= 0 || StringFind(cand_sym, StringSubstr(main_sym, 3, 3)) >= 0)
      score += 0.10;
   if(dt.hour >= 21 || dt.hour <= 1)
      score -= 0.08;
   return FXAI_Clamp(score, 0.0, 1.0);
}

double FXAI_ContextRedundancyPenalty(const string main_symbol,
                                     const string candidate_symbol,
                                     const string &selected[])
{
   double penalty = 0.0;
   int cand_cat = FXAI_ContextSymbolCategory(candidate_symbol);
   for(int i=0; i<ArraySize(selected); i++)
   {
      string picked = selected[i];
      if(StringLen(picked) <= 0) continue;
      if(StringCompare(picked, candidate_symbol, false) == 0)
         return 1.0;

      penalty = MathMax(penalty, 0.55 * FXAI_ContextSharedSymbolScore(candidate_symbol, picked));
      if(FXAI_ContextSymbolCategory(picked) == cand_cat)
         penalty = MathMax(penalty, 0.18);
   }
   penalty -= 0.10 * FXAI_ContextSharedSymbolScore(main_symbol, candidate_symbol);
   return FXAI_Clamp(penalty, 0.0, 1.0);
}

double FXAI_ContextPersistenceBonus(const string candidate_symbol,
                                    const string &selected[])
{
   for(int i=0; i<ArraySize(selected); i++)
   {
      if(StringCompare(selected[i], candidate_symbol, false) == 0)
         return 0.12;
   }
   return 0.0;
}

double FXAI_ContextIncrementalValueScore(const string main_symbol,
                                         const string candidate_symbol,
                                         const string &selected[])
{
   double shared = FXAI_ContextSharedSymbolScore(main_symbol, candidate_symbol);
   double redundancy = FXAI_ContextRedundancyPenalty(main_symbol, candidate_symbol, selected);
   return FXAI_Clamp(0.70 * shared + 0.30 * (1.0 - redundancy), 0.0, 1.0);
}

double FXAI_ContextCandidateScore(const string main_symbol,
                                  const string candidate_symbol,
                                  const string &selected[])
{
   int category = FXAI_ContextSymbolCategory(candidate_symbol);
   double score = FXAI_ContextCategoryPriority(category);
   score += 0.40 * FXAI_ContextIncrementalValueScore(main_symbol, candidate_symbol, selected);
   score += 0.24 * FXAI_ContextLiquidityScore(candidate_symbol);
   score += 0.18 * FXAI_ContextDataHealthScore(candidate_symbol);
   score += 0.14 * FXAI_ContextSessionOverlapScore(main_symbol, candidate_symbol);
   score += FXAI_ContextPersistenceBonus(candidate_symbol, selected);
   return score;
}

void FXAI_AppendUniqueContextCandidate(string &arr[],
                                       const string symbol)
{
   if(StringLen(symbol) <= 0) return;
   for(int i=0; i<ArraySize(arr); i++)
   {
      if(StringCompare(arr[i], symbol, false) == 0)
         return;
   }
   int sz = ArraySize(arr);
   ArrayResize(arr, sz + 1);
   arr[sz] = symbol;
}

void FXAI_BuildContextScoreReference(const string &selected[],
                                     const string &pending[],
                                     string &reference[])
{
   ArrayResize(reference, 0);
   for(int i=0; i<ArraySize(selected); i++)
      FXAI_AppendUniqueContextCandidate(reference, selected[i]);
   for(int i=0; i<ArraySize(pending); i++)
      FXAI_AppendUniqueContextCandidate(reference, pending[i]);
}

void FXAI_BuildCuratedContextUniverse(const string main_symbol,
                                      string &candidates[])
{
   ArrayResize(candidates, 0);
   string curated[] =
   {
      "EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF","EURJPY","GBPJPY","EURGBP",
      "XAUUSD","XAGUSD",
      "US500","SPX500","NAS100","USTEC","US30","DE40","GER40","UK100","JP225",
      "WTI","BRENT",
      "VIX","DXY","USDX","TNX","USB10","US10Y","BUND"
   };
   for(int i=0; i<ArraySize(curated); i++)
   {
      string sym = curated[i];
      if(StringCompare(sym, main_symbol, false) == 0) continue;
      if(!SymbolSelect(sym, true)) continue;
      FXAI_AppendUniqueContextCandidate(candidates, sym);
   }
}

void FXAI_ParseContextSymbols(const string raw, string &symbols[])
{
   ArrayResize(symbols, 0);

   string clean = raw;
   StringReplace(clean, "{", "");
   StringReplace(clean, "}", "");
   StringReplace(clean, ";", ",");
   StringReplace(clean, "|", ",");

   string parts[];
   int n = StringSplit(clean, ',', parts);
   if(n <= 0) return;

   for(int i=0; i<n; i++)
   {
      string sym = parts[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(StringLen(sym) <= 0) continue;

      bool exists = false;
      for(int j=0; j<ArraySize(symbols); j++)
      {
         if(StringCompare(symbols[j], sym, false) == 0)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;

      int sz = ArraySize(symbols);
      ArrayResize(symbols, sz + 1);
      symbols[sz] = sym;
      if(ArraySize(symbols) >= FXAI_MAX_CONTEXT_SYMBOLS)
         break;
   }
}

void FXAI_FilterContextSymbols(const string main_symbol, string &symbols[])
{
   int n = ArraySize(symbols);
   if(n <= 0) return;

   int w = 0;
   for(int i=0; i<n; i++)
   {
      string sym = symbols[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(StringLen(sym) <= 0) continue;
      if(StringCompare(sym, main_symbol, false) == 0) continue;
      if(!SymbolSelect(sym, true)) continue;

      symbols[w] = sym;
      w++;
   }

   if(w < n)
      ArrayResize(symbols, w);
}

void FXAI_ExtendContextSymbolsFromMarketWatch(const string main_symbol, string &symbols[])
{
   int cat_caps[FXAI_CONTEXT_CAT_COUNT];
   cat_caps[FXAI_CONTEXT_CAT_FX] = 24;
   cat_caps[FXAI_CONTEXT_CAT_METAL] = 8;
   cat_caps[FXAI_CONTEXT_CAT_INDEX] = 8;
   cat_caps[FXAI_CONTEXT_CAT_ENERGY] = 4;
   cat_caps[FXAI_CONTEXT_CAT_CRYPTO] = 2;
   cat_caps[FXAI_CONTEXT_CAT_RISK] = 8;
   cat_caps[FXAI_CONTEXT_CAT_OTHER] = 4;

   int cat_used[FXAI_CONTEXT_CAT_COUNT];
   for(int c=0; c<FXAI_CONTEXT_CAT_COUNT; c++)
      cat_used[c] = 0;
   for(int j=0; j<ArraySize(symbols); j++)
   {
      int cat = FXAI_ContextSymbolCategory(symbols[j]);
      if(cat >= 0 && cat < FXAI_CONTEXT_CAT_COUNT)
         cat_used[cat]++;
   }

   string best_sym[];
   double best_score[];
   int best_cat[];
   ArrayResize(best_sym, 0);
   ArrayResize(best_score, 0);
   ArrayResize(best_cat, 0);

   // First pass: selected Market Watch symbols. Second pass: broader terminal universe.
   for(int pass=0; pass<2; pass++)
   {
      bool selected_only = (pass == 0);
      int total = SymbolsTotal(selected_only);
      if(total <= 0) continue;
      int scan_cap = total;
      if(!selected_only && scan_cap > 384)
         scan_cap = 384;

      for(int i=0; i<scan_cap; i++)
      {
         string sym = SymbolName(i, selected_only);
         StringTrimLeft(sym);
         StringTrimRight(sym);
         if(StringLen(sym) <= 0) continue;
         if(StringCompare(sym, main_symbol, false) == 0) continue;

         bool exists = false;
         for(int j=0; j<ArraySize(symbols); j++)
         {
            if(StringCompare(symbols[j], sym, false) == 0)
            {
               exists = true;
               break;
            }
         }
         if(!exists)
         {
            for(int j=0; j<ArraySize(best_sym); j++)
            {
               if(StringCompare(best_sym[j], sym, false) == 0)
               {
                  exists = true;
                  break;
               }
            }
         }
         if(exists) continue;
         if(!SymbolSelect(sym, true)) continue;

         int cat = FXAI_ContextSymbolCategory(sym);
         if(cat < 0 || cat >= FXAI_CONTEXT_CAT_COUNT)
            cat = FXAI_CONTEXT_CAT_OTHER;
         if(cat_used[cat] >= cat_caps[cat])
            continue;

         string reference[];
         FXAI_BuildContextScoreReference(symbols, best_sym, reference);
         double score = FXAI_ContextCandidateScore(main_symbol, sym, reference);
         if(!selected_only)
         {
            // Broader-universe candidates need a slightly higher bar to justify
            // being pulled into context versus already-selected Market Watch symbols.
            score -= 0.10;
            if(cat != FXAI_CONTEXT_CAT_OTHER)
               score += 0.06;
         }
         if(score <= 0.0)
            continue;

         int sz = ArraySize(best_sym);
         ArrayResize(best_sym, sz + 1);
         ArrayResize(best_score, sz + 1);
         ArrayResize(best_cat, sz + 1);
         best_sym[sz] = sym;
         best_score[sz] = score;
         best_cat[sz] = cat;

         for(int k=sz; k>0; k--)
         {
            if(best_score[k] <= best_score[k - 1]) break;
            double tmp_score = best_score[k - 1];
            best_score[k - 1] = best_score[k];
            best_score[k] = tmp_score;
            string tmp_sym = best_sym[k - 1];
            best_sym[k - 1] = best_sym[k];
            best_sym[k] = tmp_sym;
            int tmp_cat = best_cat[k - 1];
            best_cat[k - 1] = best_cat[k];
            best_cat[k] = tmp_cat;
         }
      }
   }

   string curated[];
   FXAI_BuildCuratedContextUniverse(main_symbol, curated);
   for(int i=0; i<ArraySize(curated); i++)
   {
      string sym = curated[i];
      bool exists = false;
      for(int j=0; j<ArraySize(symbols); j++)
      {
         if(StringCompare(symbols[j], sym, false) == 0)
         {
            exists = true;
            break;
         }
      }
      if(!exists)
      {
         for(int j=0; j<ArraySize(best_sym); j++)
         {
            if(StringCompare(best_sym[j], sym, false) == 0)
            {
               exists = true;
               break;
            }
         }
      }
      if(exists) continue;

      int cat = FXAI_ContextSymbolCategory(sym);
      if(cat < 0 || cat >= FXAI_CONTEXT_CAT_COUNT)
         cat = FXAI_CONTEXT_CAT_OTHER;
      if(cat_used[cat] >= cat_caps[cat])
         continue;

      string reference[];
      FXAI_BuildContextScoreReference(symbols, best_sym, reference);
      double score = FXAI_ContextCandidateScore(main_symbol, sym, reference) + 0.08;
      if(score <= 0.0)
         continue;
      int sz = ArraySize(best_sym);
      ArrayResize(best_sym, sz + 1);
      ArrayResize(best_score, sz + 1);
      ArrayResize(best_cat, sz + 1);
      best_sym[sz] = sym;
      best_score[sz] = score;
      best_cat[sz] = cat;
      for(int k=sz; k>0; k--)
      {
         if(best_score[k] <= best_score[k - 1]) break;
         double tmp_score = best_score[k - 1];
         best_score[k - 1] = best_score[k];
         best_score[k] = tmp_score;
         string tmp_sym = best_sym[k - 1];
         best_sym[k - 1] = best_sym[k];
         best_sym[k] = tmp_sym;
         int tmp_cat = best_cat[k - 1];
         best_cat[k - 1] = best_cat[k];
         best_cat[k] = tmp_cat;
      }
   }

   for(int i=0; i<ArraySize(best_sym); i++)
   {
      int sz = ArraySize(symbols);
      if(sz >= FXAI_MAX_CONTEXT_SYMBOLS)
         break;
      int cat = best_cat[i];
      if(cat >= 0 && cat < FXAI_CONTEXT_CAT_COUNT && cat_used[cat] >= cat_caps[cat])
         continue;
      ArrayResize(symbols, sz + 1);
      symbols[sz] = best_sym[i];
      if(cat >= 0 && cat < FXAI_CONTEXT_CAT_COUNT)
         cat_used[cat]++;
   }
}

double FXAI_ContextAlignedCorr(const double &main_close[],
                               const int main_i,
                               const FXAIContextSeries &ctx,
                               const int window)
{
   int n_main = ArraySize(main_close);
   int n_map = ArraySize(ctx.aligned_idx);
   if(window < 4 || main_i < 0 || main_i >= n_main || main_i >= n_map)
      return 0.0;

   double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
   int used = 0;
   for(int k=0; k<window; k++)
   {
      int im = main_i + k;
      if(im + 1 >= n_main || im >= n_map) break;
      int ic = ctx.aligned_idx[im];
      if(ic < 0 || ic + 1 >= ArraySize(ctx.close)) continue;

      double xr = FXAI_SafeReturn(main_close, im, im + 1);
      double yr = FXAI_SafeReturn(ctx.close, ic, ic + 1);
      sx += xr;
      sy += yr;
      sxx += xr * xr;
      syy += yr * yr;
      sxy += xr * yr;
      used++;
   }

   if(used < 4) return 0.0;
   double cov = sxy - (sx * sy) / (double)used;
   double vx = sxx - (sx * sx) / (double)used;
   double vy = syy - (sy * sy) / (double)used;
   if(vx <= 1e-12 || vy <= 1e-12) return 0.0;
   return FXAI_Clamp(cov / MathSqrt(vx * vy), -1.0, 1.0);
}

void FXAI_PrecomputeContextAggregates(const datetime &main_time[],
                                     const double &main_close[],
                                     FXAIContextSeries &ctx_series[],
                                     const int ctx_count,
                                     const int upto_index,
                                     double &ctx_mean_arr[],
                                     double &ctx_std_arr[],
                                     double &ctx_up_arr[],
                                     double &ctx_extra_arr[])
{
   int n = ArraySize(main_time);
   if(ArraySize(main_close) != n) return;
   ArrayResize(ctx_mean_arr, n);
   ArrayResize(ctx_std_arr, n);
   ArrayResize(ctx_up_arr, n);
   ArrayResize(ctx_extra_arr, n * FXAI_CONTEXT_EXTRA_FEATS);
   if(n <= 0) return;

   int lag_m1 = 2 * PeriodSeconds(PERIOD_M1);
   if(lag_m1 <= 0) lag_m1 = 120;

   int upto = upto_index;
   if(upto < 0) upto = 0;
   if(upto >= n) upto = n - 1;
   // Keep one extra index ready for normalizers that need previous-bar features (i+1).
   int upto_fill = upto;
   if(upto_fill < n - 1) upto_fill++;

   for(int i=0; i<n; i++)
   {
      ctx_mean_arr[i] = 0.0;
      ctx_std_arr[i] = 0.0;
      ctx_up_arr[i] = 0.5;
   }
   for(int i=0; i<ArraySize(ctx_extra_arr); i++)
      ctx_extra_arr[i] = 0.0;

   for(int s=0; s<ctx_count; s++)
   {
      if(!ctx_series[s].loaded)
      {
         ArrayResize(ctx_series[s].aligned_idx, 0);
         continue;
      }
      FXAI_BuildAlignedIndexMapRange(main_time,
                                    ctx_series[s].time,
                                    lag_m1,
                                    upto_fill,
                                    ctx_series[s].aligned_idx);
   }

   for(int i=0; i<=upto_fill; i++)
   {
      datetime t_ref = main_time[i];
      if(t_ref <= 0 || ctx_count <= 0) continue;

      double main_ret = FXAI_SafeReturn(main_close, i, i + 1);
      double main_vol = FXAI_RollingAbsReturn(main_close, i, 20);
      if(main_vol < 1e-6) main_vol = MathAbs(main_ret);
      if(main_vol < 1e-6) main_vol = 1e-4;

      double weighted_sum = 0.0;
      double weighted_sum2 = 0.0;
      double weight_total = 0.0;
      double up_weight = 0.0;
      int valid = 0;
      double top_score[FXAI_CONTEXT_TOP_SYMBOLS];
      int top_symbol_idx[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_ret[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_lag[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_rel[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_corr[FXAI_CONTEXT_TOP_SYMBOLS];
      for(int t=0; t<FXAI_CONTEXT_TOP_SYMBOLS; t++)
      {
         top_score[t] = -1e18;
         top_symbol_idx[t] = -1;
         top_ctx_ret[t] = 0.0;
         top_ctx_lag[t] = 0.0;
         top_ctx_rel[t] = 0.0;
         top_ctx_corr[t] = 0.0;
      }

      for(int s=0; s<ctx_count; s++)
      {
         if(!ctx_series[s].loaded) continue;
         int idx = -1;
         if(i >= 0 && i < ArraySize(ctx_series[s].aligned_idx))
            idx = ctx_series[s].aligned_idx[i];
         if(idx < 0) continue;

         double freshness = FXAI_AlignedFreshnessWeight(ctx_series[s].time, idx, t_ref, lag_m1);
         double ctx_ret_raw = FXAI_SafeReturn(ctx_series[s].close, idx, idx + 1);
         double ctx_lag_raw = FXAI_SafeReturn(ctx_series[s].close, idx + 1, idx + 2);
         double ctx_rel_raw = ctx_ret_raw - main_ret;
         double ctx_corr_raw = FXAI_ContextAlignedCorr(main_close, i, ctx_series[s], 20);
         double rel_edge = FXAI_Clamp(MathAbs(ctx_rel_raw) / main_vol, 0.0, 4.0);
         double ret_edge = FXAI_Clamp(MathAbs(ctx_ret_raw) / main_vol, 0.0, 4.0);
         double lag_edge = FXAI_Clamp(MathAbs(ctx_lag_raw) / main_vol, 0.0, 4.0);
         double corr_edge = MathAbs(ctx_corr_raw);
         double symbol_score = freshness * ((0.40 * corr_edge) +
                                            (0.30 * rel_edge) +
                                            (0.20 * lag_edge) +
                                            (0.10 * ret_edge));

         double w = 0.20 + symbol_score;
         weighted_sum += w * ctx_ret_raw;
         weighted_sum2 += w * ctx_ret_raw * ctx_ret_raw;
         weight_total += w;
         if(ctx_ret_raw > 0.0) up_weight += w;
         valid++;

         for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
         {
            if(symbol_score <= top_score[slot]) continue;
            for(int shift=FXAI_CONTEXT_TOP_SYMBOLS - 1; shift>slot; shift--)
            {
               top_score[shift] = top_score[shift - 1];
               top_symbol_idx[shift] = top_symbol_idx[shift - 1];
               top_ctx_ret[shift] = top_ctx_ret[shift - 1];
               top_ctx_lag[shift] = top_ctx_lag[shift - 1];
               top_ctx_rel[shift] = top_ctx_rel[shift - 1];
               top_ctx_corr[shift] = top_ctx_corr[shift - 1];
            }
            top_score[slot] = symbol_score;
            top_symbol_idx[slot] = s;
            top_ctx_ret[slot] = ctx_ret_raw * freshness;
            top_ctx_lag[slot] = ctx_lag_raw * freshness;
            top_ctx_rel[slot] = ctx_rel_raw * freshness;
            top_ctx_corr[slot] = ctx_corr_raw * freshness;
            break;
         }
      }

      if(valid <= 0 || weight_total <= 0.0) continue;

      double mean = weighted_sum / weight_total;
      double var = (weighted_sum2 / weight_total) - (mean * mean);
      if(var < 0.0) var = 0.0;
      double up_ratio = up_weight / weight_total;

      double coverage = (ctx_count > 0 ? ((double)valid / (double)ctx_count) : 0.0);
      coverage = FXAI_Clamp(coverage, 0.0, 1.0);
      double conf = 0.30 + (0.70 * coverage);

      ctx_mean_arr[i] = mean * coverage;
      ctx_std_arr[i] = MathSqrt(var) * conf;
      ctx_up_arr[i] = 0.5 + ((up_ratio - 0.5) * coverage);

      for(int top_slot=0; top_slot<FXAI_CONTEXT_TOP_SYMBOLS; top_slot++)
      {
         if(top_symbol_idx[top_slot] < 0) continue;
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 0, top_ctx_ret[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 1, top_ctx_lag[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 2, top_ctx_rel[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 3, top_ctx_corr[top_slot]);
         int top_idx = (i >= 0 && i < ArraySize(ctx_series[top_symbol_idx[top_slot]].aligned_idx)
                        ? ctx_series[top_symbol_idx[top_slot]].aligned_idx[i]
                        : -1);
         FXAI_SetContextSlotMTFStateExtras(ctx_extra_arr,
                                           i,
                                           top_slot,
                                           ctx_series[top_symbol_idx[top_slot]],
                                           top_idx);
      }

      double adapter_stability = 0.5;
      double adapter_lead = 0.5;
      int adapter_used = 0;
      for(int top_slot=0; top_slot<FXAI_CONTEXT_TOP_SYMBOLS; top_slot++)
      {
         if(top_symbol_idx[top_slot] < 0) continue;
         double stab = 1.0 - FXAI_Clamp(MathAbs(top_ctx_ret[top_slot] - top_ctx_lag[top_slot]) / MathMax(main_vol, 1e-4), 0.0, 1.0);
         double lead = FXAI_Clamp(MathAbs(top_ctx_lag[top_slot]) / MathMax(main_vol, 1e-4), 0.0, 4.0) / 4.0;
         adapter_stability += stab;
         adapter_lead += lead;
         adapter_used++;
      }
      if(adapter_used > 0)
      {
         adapter_stability /= (double)(adapter_used + 1);
         adapter_lead /= (double)(adapter_used + 1);
      }

      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 0, FXAI_Clamp(mean / MathMax(main_vol, 1e-4), -1.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 1, FXAI_Clamp(adapter_stability, 0.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 2, FXAI_Clamp(adapter_lead, 0.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 3, coverage);
   }
}

double FXAI_ContextSeriesUtilityAt(const datetime &main_time[],
                                   const double &main_close[],
                                   FXAIContextSeries &series,
                                   const int main_i)
{
   if(!series.loaded) return -1e9;
   if(ArraySize(main_time) <= 4 || ArraySize(main_close) <= 4) return -1e9;
   if(ArraySize(series.aligned_idx) <= 4) return -1e9;
   if(main_i < 0 || main_i >= ArraySize(main_time) || main_i >= ArraySize(series.aligned_idx))
      return -1e9;

   double sum_score = 0.0;
   int used = 0;
   for(int i=0; i<16; i++)
   {
      int im = main_i + i;
      if(im >= ArraySize(main_time) || im >= ArraySize(series.aligned_idx)) break;
      int idx = series.aligned_idx[im];
      if(idx < 0) continue;

      double lag = 2.0 * PeriodSeconds(PERIOD_M1);
      if(lag <= 0.0) lag = 120.0;
      double fresh = FXAI_AlignedFreshnessWeight(series.time, idx, main_time[im], (int)lag);
      double main_ret = FXAI_SafeReturn(main_close, im, im + 1);
      double ctx_ret = FXAI_SafeReturn(series.close, idx, idx + 1);
      double ctx_lag = FXAI_SafeReturn(series.close, idx + 1, idx + 2);
      double corr = FXAI_ContextAlignedCorr(main_close, im, series, 20);
      double vol = FXAI_RollingAbsReturn(main_close, im, 20);
      if(vol < 1e-6) vol = MathAbs(main_ret);
      if(vol < 1e-6) vol = 1e-4;

      double rel = FXAI_Clamp(MathAbs(ctx_ret - main_ret) / vol, 0.0, 4.0);
      double lead = FXAI_Clamp(MathAbs(ctx_lag) / vol, 0.0, 4.0);
      double mag = FXAI_Clamp(MathAbs(ctx_ret) / vol, 0.0, 4.0);
      double score = fresh * ((0.45 * MathAbs(corr)) +
                              (0.30 * rel) +
                              (0.15 * lead) +
                              (0.10 * mag));
      sum_score += score;
      used++;
   }

   if(used <= 0) return -1e9;
   return sum_score / (double)used;
}

void FXAI_SetContextSlotMTFStateExtras(double &ctx_extra_arr[],
                                       const int sample_idx,
                                       const int top_slot,
                                       FXAIContextSeries &series,
                                       const int idx)
{
   if(top_slot < 0 || top_slot >= FXAI_CONTEXT_TOP_SYMBOLS)
      return;
   if(!series.loaded || idx < 0)
      return;

   double point_value = SymbolInfoDouble(series.symbol, SYMBOL_POINT);
   if(point_value <= 0.0)
      point_value = (_Point > 0.0 ? _Point : 1.0);

   for(int tf_slot=0; tf_slot<FXAI_CONTEXT_MTF_TF_COUNT; tf_slot++)
   {
      double body_bias = 0.0;
      double close_loc = 0.0;
      double range_pressure = 0.0;
      double spread_pressure = 0.0;
      int bars = FXAI_ContextMTFBarsForSlot(tf_slot);
      if(!FXAI_ComputeAggregatedCandleSpreadState(idx,
                                                  bars,
                                                  series.open,
                                                  series.high,
                                                  series.low,
                                                  series.close,
                                                  series.spread,
                                                  point_value,
                                                  body_bias,
                                                  close_loc,
                                                  range_pressure,
                                                  spread_pressure))
         continue;

      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_BODY_BIAS), body_bias);
      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_CLOSE_LOCATION), close_loc);
      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_RANGE_PRESSURE), range_pressure);
      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_SPREAD_PRESSURE), spread_pressure);
   }
}

void FXAI_PrecomputeDynamicContextAggregates(const datetime &main_time[],
                                             const double &main_close[],
                                             FXAIContextSeries &ctx_series[],
                                             const int ctx_count,
                                             const int upto_index,
                                             double &ctx_mean_arr[],
                                             double &ctx_std_arr[],
                                             double &ctx_up_arr[],
                                             double &ctx_extra_arr[])
{
   int n = ArraySize(main_time);
   if(ArraySize(main_close) != n) return;
   ArrayResize(ctx_mean_arr, n);
   ArrayResize(ctx_std_arr, n);
   ArrayResize(ctx_up_arr, n);
   ArrayResize(ctx_extra_arr, n * FXAI_CONTEXT_EXTRA_FEATS);
   if(n <= 0) return;

   int lag_m1 = 2 * PeriodSeconds(PERIOD_M1);
   if(lag_m1 <= 0) lag_m1 = 120;

   int upto = upto_index;
   if(upto < 0) upto = 0;
   if(upto >= n) upto = n - 1;
   int upto_fill = upto;
   if(upto_fill < n - 1) upto_fill++;

   for(int i=0; i<n; i++)
   {
      ctx_mean_arr[i] = 0.0;
      ctx_std_arr[i] = 0.0;
      ctx_up_arr[i] = 0.5;
   }
   for(int i=0; i<ArraySize(ctx_extra_arr); i++)
      ctx_extra_arr[i] = 0.0;

   double local_util[FXAI_MAX_CONTEXT_SYMBOLS];
   double local_stability[FXAI_MAX_CONTEXT_SYMBOLS];
   double local_lead[FXAI_MAX_CONTEXT_SYMBOLS];
   double local_coverage[FXAI_MAX_CONTEXT_SYMBOLS];
   bool local_ready[FXAI_MAX_CONTEXT_SYMBOLS];
   for(int s=0; s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
   {
      local_util[s] = 0.0;
      local_stability[s] = 0.0;
      local_lead[s] = 0.0;
      local_coverage[s] = 0.0;
      local_ready[s] = false;
   }

   for(int s=0; s<ctx_count; s++)
   {
      if(!ctx_series[s].loaded)
      {
         ArrayResize(ctx_series[s].aligned_idx, 0);
         continue;
      }
      FXAI_BuildAlignedIndexMapRange(main_time,
                                    ctx_series[s].time,
                                    lag_m1,
                                    upto_fill,
                                    ctx_series[s].aligned_idx);
   }

   for(int i=upto_fill; i>=0; i--)
   {
      datetime t_ref = main_time[i];
      if(t_ref <= 0 || ctx_count <= 0) continue;

      double main_ret = FXAI_SafeReturn(main_close, i, i + 1);
      double main_vol = FXAI_RollingAbsReturn(main_close, i, 20);
      if(main_vol < 1e-6) main_vol = MathAbs(main_ret);
      if(main_vol < 1e-6) main_vol = 1e-4;

      double select_score[FXAI_MAX_CONTEXT_SYMBOLS];
      int select_idx[FXAI_MAX_CONTEXT_SYMBOLS];
      int keep_n = ctx_count;
      if(keep_n > FXAI_CONTEXT_DYNAMIC_POOL) keep_n = FXAI_CONTEXT_DYNAMIC_POOL;
      if(main_vol < 0.00025 && keep_n > 4) keep_n = 4;
      if(main_vol > 0.00120 && keep_n < FXAI_CONTEXT_DYNAMIC_POOL)
         keep_n++;
      for(int s=0; s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
      {
         select_score[s] = -1e18;
         select_idx[s] = -1;
      }

      for(int s=0; s<ctx_count && s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
      {
         if(!ctx_series[s].loaded) continue;
         double util = FXAI_ContextSeriesUtilityAt(main_time, main_close, ctx_series[s], i);
         if(util <= -1e8) continue;

         int idx_prior = (i < ArraySize(ctx_series[s].aligned_idx) ? ctx_series[s].aligned_idx[i] : -1);
         if(idx_prior < 0) continue;
         double freshness_prior = FXAI_AlignedFreshnessWeight(ctx_series[s].time, idx_prior, t_ref, lag_m1);
         double ctx_ret_prior = FXAI_SafeReturn(ctx_series[s].close, idx_prior, idx_prior + 1);
         double ctx_lag_prior = FXAI_SafeReturn(ctx_series[s].close, idx_prior + 1, idx_prior + 2);
         double ctx_corr_prior = FXAI_ContextAlignedCorr(main_close, i, ctx_series[s], 20);
         double stability_local = 1.0 - FXAI_Clamp(MathAbs(ctx_ret_prior - ctx_lag_prior) / MathMax(main_vol, 1e-4), 0.0, 1.0);
         double lead_local = FXAI_Clamp(MathAbs(ctx_lag_prior) / MathMax(main_vol, 1e-4), 0.0, 4.0) / 4.0;
         double coverage_local = freshness_prior;
         double prior_util = (local_ready[s] ? local_util[s] : util);
         double prior_stability = (local_ready[s] ? local_stability[s] : stability_local);
         double prior_lead = (local_ready[s] ? local_lead[s] : lead_local);
         double prior_coverage = (local_ready[s] ? local_coverage[s] : coverage_local);
         double rank_score = 0.48 * util +
                             0.18 * FXAI_Clamp(ctx_corr_prior, -1.0, 1.0) +
                             0.12 * prior_util +
                             0.10 * prior_stability +
                             0.07 * prior_lead +
                             0.05 * prior_coverage;

         for(int slot=0; slot<keep_n; slot++)
         {
            if(rank_score <= select_score[slot]) continue;
            for(int shift=keep_n - 1; shift>slot; shift--)
            {
               select_score[shift] = select_score[shift - 1];
               select_idx[shift] = select_idx[shift - 1];
            }
            select_score[slot] = rank_score;
            select_idx[slot] = s;
            break;
         }
      }

      double weighted_sum = 0.0;
      double weighted_sum2 = 0.0;
      double weight_total = 0.0;
      double up_weight = 0.0;
      double stability_sum = 0.0;
      double lead_sum = 0.0;
      int valid = 0;
      double top_score[FXAI_CONTEXT_TOP_SYMBOLS];
      int top_symbol_idx[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_ret[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_lag[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_rel[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_corr[FXAI_CONTEXT_TOP_SYMBOLS];
      for(int t=0; t<FXAI_CONTEXT_TOP_SYMBOLS; t++)
      {
         top_score[t] = -1e18;
         top_symbol_idx[t] = -1;
         top_ctx_ret[t] = 0.0;
         top_ctx_lag[t] = 0.0;
         top_ctx_rel[t] = 0.0;
         top_ctx_corr[t] = 0.0;
      }

      for(int slot=0; slot<keep_n; slot++)
      {
         int s = select_idx[slot];
         if(s < 0 || s >= ctx_count) continue;
         if(!ctx_series[s].loaded) continue;
         if(i >= ArraySize(ctx_series[s].aligned_idx)) continue;

         int idx = ctx_series[s].aligned_idx[i];
         if(idx < 0) continue;

         double freshness = FXAI_AlignedFreshnessWeight(ctx_series[s].time, idx, t_ref, lag_m1);
         double ctx_ret_raw = FXAI_SafeReturn(ctx_series[s].close, idx, idx + 1);
         double ctx_lag_raw = FXAI_SafeReturn(ctx_series[s].close, idx + 1, idx + 2);
         double ctx_rel_raw = ctx_ret_raw - main_ret;
         double ctx_corr_raw = FXAI_ContextAlignedCorr(main_close, i, ctx_series[s], 20);
         double rel_edge = FXAI_Clamp(MathAbs(ctx_rel_raw) / main_vol, 0.0, 4.0);
         double ret_edge = FXAI_Clamp(MathAbs(ctx_ret_raw) / main_vol, 0.0, 4.0);
         double lag_edge = FXAI_Clamp(MathAbs(ctx_lag_raw) / main_vol, 0.0, 4.0);
         double corr_edge = MathAbs(ctx_corr_raw);
         double stability_edge = 1.0 - FXAI_Clamp(MathAbs(ctx_ret_raw - ctx_lag_raw) / MathMax(main_vol, 1e-4), 0.0, 1.0);
         double prior_util = (local_ready[s] ? local_util[s] : 0.0);
         double prior_stability = (local_ready[s] ? local_stability[s] : stability_edge);
         double symbol_score = freshness * ((0.28 * corr_edge) +
                                            (0.24 * rel_edge) +
                                            (0.18 * lag_edge) +
                                            (0.12 * ret_edge) +
                                            (0.10 * stability_edge) +
                                            (0.08 * FXAI_Clamp(prior_util, -1.0, 1.0)));
         symbol_score += 0.05 * prior_stability;

         double w = 0.20 + symbol_score;
         weighted_sum += w * ctx_ret_raw;
         weighted_sum2 += w * ctx_ret_raw * ctx_ret_raw;
         weight_total += w;
         if(ctx_ret_raw > 0.0) up_weight += w;
         stability_sum += w * stability_edge;
         lead_sum += w * (lag_edge / 4.0);
         valid++;

         for(int top_slot=0; top_slot<FXAI_CONTEXT_TOP_SYMBOLS; top_slot++)
         {
            if(symbol_score <= top_score[top_slot]) continue;
            for(int shift=FXAI_CONTEXT_TOP_SYMBOLS - 1; shift>top_slot; shift--)
            {
               top_score[shift] = top_score[shift - 1];
               top_symbol_idx[shift] = top_symbol_idx[shift - 1];
               top_ctx_ret[shift] = top_ctx_ret[shift - 1];
               top_ctx_lag[shift] = top_ctx_lag[shift - 1];
               top_ctx_rel[shift] = top_ctx_rel[shift - 1];
               top_ctx_corr[shift] = top_ctx_corr[shift - 1];
            }
            top_score[top_slot] = symbol_score;
            top_symbol_idx[top_slot] = s;
            top_ctx_ret[top_slot] = ctx_ret_raw * freshness;
            top_ctx_lag[top_slot] = ctx_lag_raw * freshness;
            top_ctx_rel[top_slot] = ctx_rel_raw * freshness;
            top_ctx_corr[top_slot] = ctx_corr_raw * freshness;
            break;
         }
      }

      if(valid <= 0 || weight_total <= 0.0) continue;

      double mean = weighted_sum / weight_total;
      double var = (weighted_sum2 / weight_total) - (mean * mean);
      if(var < 0.0) var = 0.0;
      double up_ratio = up_weight / weight_total;
      double coverage = (keep_n > 0 ? ((double)valid / (double)keep_n) : 0.0);
      coverage = FXAI_Clamp(coverage, 0.0, 1.0);
      double conf = 0.30 + (0.70 * coverage);

      ctx_mean_arr[i] = mean * coverage;
      ctx_std_arr[i] = MathSqrt(var) * conf;
      ctx_up_arr[i] = 0.5 + ((up_ratio - 0.5) * coverage);

      for(int top_slot=0; top_slot<FXAI_CONTEXT_TOP_SYMBOLS; top_slot++)
      {
         if(top_symbol_idx[top_slot] < 0) continue;
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 0, top_ctx_ret[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 1, top_ctx_lag[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 2, top_ctx_rel[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 3, top_ctx_corr[top_slot]);
         int top_idx = (i >= 0 && i < ArraySize(ctx_series[top_symbol_idx[top_slot]].aligned_idx)
                        ? ctx_series[top_symbol_idx[top_slot]].aligned_idx[i]
                        : -1);
         FXAI_SetContextSlotMTFStateExtras(ctx_extra_arr,
                                           i,
                                           top_slot,
                                           ctx_series[top_symbol_idx[top_slot]],
                                           top_idx);
      }

      double adapter_util = FXAI_Clamp(mean / MathMax(main_vol, 1e-4), -1.0, 1.0);
      double adapter_stability = (weight_total > 0.0 ? stability_sum / weight_total : 0.5);
      double adapter_lead = (weight_total > 0.0 ? lead_sum / weight_total : 0.5);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 0, adapter_util);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 1, FXAI_Clamp(adapter_stability, 0.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 2, FXAI_Clamp(adapter_lead, 0.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 3, coverage);

      for(int s=0; s<ctx_count && s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
      {
         if(!ctx_series[s].loaded) continue;
         if(i >= ArraySize(ctx_series[s].aligned_idx)) continue;

         int idx_prior = ctx_series[s].aligned_idx[i];
         if(idx_prior < 0) continue;

         double util = FXAI_ContextSeriesUtilityAt(main_time, main_close, ctx_series[s], i);
         if(util <= -1e8) continue;

         double freshness_prior = FXAI_AlignedFreshnessWeight(ctx_series[s].time, idx_prior, t_ref, lag_m1);
         double ctx_ret_prior = FXAI_SafeReturn(ctx_series[s].close, idx_prior, idx_prior + 1);
         double ctx_lag_prior = FXAI_SafeReturn(ctx_series[s].close, idx_prior + 1, idx_prior + 2);
         double stability_local = 1.0 - FXAI_Clamp(MathAbs(ctx_ret_prior - ctx_lag_prior) / MathMax(main_vol, 1e-4), 0.0, 1.0);
         double lead_local = FXAI_Clamp(MathAbs(ctx_lag_prior) / MathMax(main_vol, 1e-4), 0.0, 4.0) / 4.0;
         double coverage_local = freshness_prior;

         if(!local_ready[s])
         {
            local_util[s] = util;
            local_stability[s] = stability_local;
            local_lead[s] = lead_local;
            local_coverage[s] = coverage_local;
            local_ready[s] = true;
         }
         else
         {
            local_util[s] = 0.85 * local_util[s] + 0.15 * util;
            local_stability[s] = 0.85 * local_stability[s] + 0.15 * stability_local;
            local_lead[s] = 0.85 * local_lead[s] + 0.15 * lead_local;
            local_coverage[s] = 0.85 * local_coverage[s] + 0.15 * coverage_local;
         }
      }
   }

   for(int s=0; s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
   {
      g_context_symbol_utility[s] = local_util[s];
      g_context_symbol_stability[s] = local_stability[s];
      g_context_symbol_lead[s] = local_lead[s];
      g_context_symbol_coverage[s] = local_coverage[s];
      g_context_symbol_utility_ready[s] = local_ready[s];
   }
}

void FXAI_GetDynamicContextState(double &utility_out,
                                 double &stability_out,
                                 double &lead_out,
                                 double &coverage_out)
{
   utility_out = 0.0;
   stability_out = 0.0;
   lead_out = 0.0;
   coverage_out = 0.0;

   double util_sum = 0.0;
   double stab_sum = 0.0;
   double lead_sum = 0.0;
   double cov_sum = 0.0;
   int used = 0;

   for(int s=0; s<ArraySize(g_context_symbols) && s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
   {
      if(!g_context_symbol_utility_ready[s]) continue;
      util_sum += g_context_symbol_utility[s];
      stab_sum += g_context_symbol_stability[s];
      lead_sum += g_context_symbol_lead[s];
      cov_sum += g_context_symbol_coverage[s];
      used++;
   }

   if(used <= 0) return;
   utility_out = util_sum / (double)used;
   stability_out = stab_sum / (double)used;
   lead_out = lead_sum / (double)used;
   coverage_out = cov_sum / (double)used;
}

//--------------------------- INIT -----------------------------------
