#ifndef __FXAI_RUNTIME_TRADE_PAIR_NETWORK_MQH__
#define __FXAI_RUNTIME_TRADE_PAIR_NETWORK_MQH__

#define FXAI_PAIR_NETWORK_MAX_REASONS 12
#define FXAI_PAIR_NETWORK_MAX_CURRENCIES 24
#define FXAI_PAIR_NETWORK_MAX_CURRENCY_PROFILES 24
#define FXAI_PAIR_NETWORK_FACTOR_COUNT 7
#define FXAI_PAIR_NETWORK_MAX_PEERS 64

#define FXAI_PAIR_NETWORK_DECISION_ALLOW 0
#define FXAI_PAIR_NETWORK_DECISION_ALLOW_REDUCED 1
#define FXAI_PAIR_NETWORK_DECISION_SUPPRESS_REDUNDANT 2
#define FXAI_PAIR_NETWORK_DECISION_BLOCK_CONTRADICTORY 3
#define FXAI_PAIR_NETWORK_DECISION_BLOCK_CONCENTRATION 4
#define FXAI_PAIR_NETWORK_DECISION_PREFER_ALTERNATIVE 5

struct FXAIPairNetworkCurrencyProfile
{
   string currency;
   double usd_bloc;
   double eur_rates;
   double safe_haven;
   double commodity_fx;
   double risk_on;
   double liquidity_stress;
   double macro_shock;
};

struct FXAIPairNetworkConfig
{
   bool     ready;
   bool     enabled;
   bool     fallback_structural_only;
   bool     auto_apply;
   bool     fallback_graph_used;
   bool     partial_dependency_data;
   bool     graph_stale;
   datetime generated_at;
   string   graph_mode;
   int      graph_stale_after_sec;
   int      history_points;
   int      max_edges_per_pair;
   int      min_empirical_overlap;
   int      empirical_lookback_bars;
   double   structural_weight;
   double   empirical_weight;
   double   redundancy_threshold;
   double   contradiction_threshold;
   double   concentration_reduce_threshold;
   double   concentration_block_threshold;
   double   execution_overlap_threshold;
   double   reduced_size_multiplier_floor;
   double   preferred_expression_margin;
   double   min_incremental_edge_score;
   double   weight_edge_after_costs;
   double   weight_execution_quality;
   double   weight_calibration_quality;
   double   weight_portfolio_fit;
   double   weight_diversification;
   double   weight_macro_fit;
   int      currency_profile_count;
   FXAIPairNetworkCurrencyProfile currency_profiles[FXAI_PAIR_NETWORK_MAX_CURRENCY_PROFILES];
};

struct FXAIPairNetworkPeerCandidate
{
   bool   ready;
   string symbol;
   int    direction;
   double size_units;
   double quality;
};

struct FXAIPairNetworkDecisionState
{
   bool     ready;
   bool     fallback_graph_used;
   bool     partial_dependency_data;
   bool     graph_stale;
   datetime generated_at;
   string   symbol;
   int      direction;
   string   decision;
   double   conflict_score;
   double   redundancy_score;
   double   contradiction_score;
   double   concentration_score;
   double   currency_concentration;
   double   factor_concentration;
   double   recommended_size_multiplier;
   string   preferred_expression;
   string   currency_exposure_csv;
   string   factor_exposure_csv;
   int      reason_count;
   string   reasons[FXAI_PAIR_NETWORK_MAX_REASONS];
};

FXAIPairNetworkConfig g_pair_network_cfg_cache;
datetime g_pair_network_cfg_cache_loaded_at = 0;
bool     g_pair_network_last_ready = false;
bool     g_pair_network_last_fallback_graph_used = false;
bool     g_pair_network_last_partial_dependency_data = false;
bool     g_pair_network_last_graph_stale = true;
datetime g_pair_network_last_generated_at = 0;
string   g_pair_network_last_symbol = "";
string   g_pair_network_last_decision = "ALLOW";
double   g_pair_network_last_conflict_score = 0.0;
double   g_pair_network_last_redundancy_score = 0.0;
double   g_pair_network_last_contradiction_score = 0.0;
double   g_pair_network_last_concentration_score = 0.0;
double   g_pair_network_last_currency_concentration = 0.0;
double   g_pair_network_last_factor_concentration = 0.0;
double   g_pair_network_last_recommended_size_multiplier = 1.0;
string   g_pair_network_last_preferred_expression = "";
string   g_pair_network_last_reasons_csv = "";

string FXAI_PairNetworkConfigFile(void)
{
   return "FXAI\\Runtime\\pair_network_config.tsv";
}

string FXAI_PairNetworkStatusFile(void)
{
   return "FXAI\\Runtime\\pair_network_status.tsv";
}

string FXAI_PairNetworkRuntimeStateFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_pair_network_" + FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_PairNetworkRuntimeHistoryFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_pair_network_history_" + FXAI_ControlPlaneSafeToken(symbol) + ".ndjson";
}

string FXAI_PairNetworkISO8601(const datetime value)
{
   if(value <= 0)
      return "";
   MqlDateTime dt;
   TimeToStruct(value, dt);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",
                       dt.year,
                       dt.mon,
                       dt.day,
                       dt.hour,
                       dt.min,
                       dt.sec);
}

string FXAI_PairNetworkJSONEscape(const string raw)
{
   string out = raw;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", " ");
   StringReplace(out, "\n", " ");
   return out;
}

void FXAI_ResetPairNetworkCurrencyProfile(FXAIPairNetworkCurrencyProfile &out)
{
   out.currency = "";
   out.usd_bloc = 0.0;
   out.eur_rates = 0.0;
   out.safe_haven = 0.0;
   out.commodity_fx = 0.0;
   out.risk_on = 0.0;
   out.liquidity_stress = 0.0;
   out.macro_shock = 0.20;
}

void FXAI_ResetPairNetworkConfig(FXAIPairNetworkConfig &out)
{
   out.ready = true;
   out.enabled = true;
   out.fallback_structural_only = true;
   out.auto_apply = true;
   out.fallback_graph_used = false;
   out.partial_dependency_data = false;
   out.graph_stale = true;
   out.generated_at = 0;
   out.graph_mode = "STRUCTURAL_ONLY";
   out.graph_stale_after_sec = 43200;
   out.history_points = 192;
   out.max_edges_per_pair = 10;
   out.min_empirical_overlap = 128;
   out.empirical_lookback_bars = 512;
   out.structural_weight = 0.72;
   out.empirical_weight = 0.28;
   out.redundancy_threshold = 0.68;
   out.contradiction_threshold = 0.74;
   out.concentration_reduce_threshold = 0.58;
   out.concentration_block_threshold = 0.80;
   out.execution_overlap_threshold = 0.62;
   out.reduced_size_multiplier_floor = 0.45;
   out.preferred_expression_margin = 0.04;
   out.min_incremental_edge_score = 0.12;
   out.weight_edge_after_costs = 0.34;
   out.weight_execution_quality = 0.20;
   out.weight_calibration_quality = 0.16;
   out.weight_portfolio_fit = 0.14;
   out.weight_diversification = 0.10;
   out.weight_macro_fit = 0.06;
   out.currency_profile_count = 0;
   for(int i=0; i<FXAI_PAIR_NETWORK_MAX_CURRENCY_PROFILES; i++)
      FXAI_ResetPairNetworkCurrencyProfile(out.currency_profiles[i]);
}

void FXAI_ResetPairNetworkDecisionState(FXAIPairNetworkDecisionState &out)
{
   out.ready = false;
   out.fallback_graph_used = false;
   out.partial_dependency_data = false;
   out.graph_stale = true;
   out.generated_at = 0;
   out.symbol = "";
   out.direction = -1;
   out.decision = "ALLOW";
   out.conflict_score = 0.0;
   out.redundancy_score = 0.0;
   out.contradiction_score = 0.0;
   out.concentration_score = 0.0;
   out.currency_concentration = 0.0;
   out.factor_concentration = 0.0;
   out.recommended_size_multiplier = 1.0;
   out.preferred_expression = "";
   out.currency_exposure_csv = "";
   out.factor_exposure_csv = "";
   out.reason_count = 0;
   for(int i=0; i<FXAI_PAIR_NETWORK_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_ResetPairNetworkGlobals(void)
{
   g_pair_network_last_ready = false;
   g_pair_network_last_fallback_graph_used = false;
   g_pair_network_last_partial_dependency_data = false;
   g_pair_network_last_graph_stale = true;
   g_pair_network_last_generated_at = 0;
   g_pair_network_last_symbol = "";
   g_pair_network_last_decision = "ALLOW";
   g_pair_network_last_conflict_score = 0.0;
   g_pair_network_last_redundancy_score = 0.0;
   g_pair_network_last_contradiction_score = 0.0;
   g_pair_network_last_concentration_score = 0.0;
   g_pair_network_last_currency_concentration = 0.0;
   g_pair_network_last_factor_concentration = 0.0;
   g_pair_network_last_recommended_size_multiplier = 1.0;
   g_pair_network_last_preferred_expression = "";
   g_pair_network_last_reasons_csv = "";
}

void FXAI_PairNetworkAppendReason(FXAIPairNetworkDecisionState &state,
                                  const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_PAIR_NETWORK_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_PairNetworkReasonsCSV(const FXAIPairNetworkDecisionState &state)
{
   string joined = "";
   for(int i=0; i<state.reason_count; i++)
   {
      if(StringLen(state.reasons[i]) <= 0)
         continue;
      if(StringLen(joined) > 0)
         joined += "; ";
      joined += state.reasons[i];
   }
   return joined;
}

string FXAI_PairNetworkFactorName(const int idx)
{
   switch(idx)
   {
      case 0: return "usd_bloc";
      case 1: return "eur_rates";
      case 2: return "safe_haven";
      case 3: return "commodity_fx";
      case 4: return "risk_on";
      case 5: return "liquidity_stress";
      case 6: return "macro_shock";
   }
   return "unknown";
}

int FXAI_PairNetworkFindCurrencyProfile(const FXAIPairNetworkConfig &cfg,
                                        const string currency)
{
   for(int i=0; i<cfg.currency_profile_count; i++)
   {
      if(cfg.currency_profiles[i].currency == currency)
         return i;
   }
   return -1;
}

double FXAI_PairNetworkProfileFactorValue(const FXAIPairNetworkCurrencyProfile &profile,
                                          const int factor_idx)
{
   switch(factor_idx)
   {
      case 0: return profile.usd_bloc;
      case 1: return profile.eur_rates;
      case 2: return profile.safe_haven;
      case 3: return profile.commodity_fx;
      case 4: return profile.risk_on;
      case 5: return profile.liquidity_stress;
      case 6: return profile.macro_shock;
   }
   return 0.0;
}

void FXAI_PairNetworkSetProfileFactorValue(FXAIPairNetworkCurrencyProfile &profile,
                                           const int factor_idx,
                                           const double value)
{
   switch(factor_idx)
   {
      case 0: profile.usd_bloc = value; break;
      case 1: profile.eur_rates = value; break;
      case 2: profile.safe_haven = value; break;
      case 3: profile.commodity_fx = value; break;
      case 4: profile.risk_on = value; break;
      case 5: profile.liquidity_stress = value; break;
      case 6: profile.macro_shock = value; break;
   }
}

double FXAI_PairNetworkCurrencyFactorWeight(const FXAIPairNetworkConfig &cfg,
                                            const string currency,
                                            const int factor_idx)
{
   int idx = FXAI_PairNetworkFindCurrencyProfile(cfg, currency);
   if(idx < 0)
      return 0.0;
   return FXAI_PairNetworkProfileFactorValue(cfg.currency_profiles[idx], factor_idx);
}

int FXAI_PairNetworkFindKey(const string &keys[],
                            const int count,
                            const string key)
{
   for(int i=0; i<count; i++)
   {
      if(keys[i] == key)
         return i;
   }
   return -1;
}

void FXAI_PairNetworkAddKeyValue(string &keys[],
                                 double &values[],
                                 int &count,
                                 const string key,
                                 const double delta,
                                 const int max_count)
{
   if(StringLen(key) <= 0 || count < 0)
      return;
   int idx = FXAI_PairNetworkFindKey(keys, count, key);
   if(count >= max_count && idx < 0)
      return;
   if(idx < 0)
   {
      idx = count;
      keys[idx] = key;
      values[idx] = 0.0;
      count++;
   }
   values[idx] += delta;
}

double FXAI_PairNetworkDot(const string &lhs_keys[],
                           const double &lhs_values[],
                           const int lhs_count,
                           const string &rhs_keys[],
                           const double &rhs_values[],
                           const int rhs_count)
{
   double dot = 0.0;
   for(int i=0; i<lhs_count; i++)
   {
      int idx = FXAI_PairNetworkFindKey(rhs_keys, rhs_count, lhs_keys[i]);
      if(idx < 0)
         continue;
      dot += lhs_values[i] * rhs_values[idx];
   }
   return dot;
}

double FXAI_PairNetworkNorm(const double &values[],
                            const int count)
{
   double total = 0.0;
   for(int i=0; i<count; i++)
      total += values[i] * values[i];
   if(total <= 0.0)
      return 0.0;
   return MathSqrt(total);
}

double FXAI_PairNetworkFactorNorm(const double &values[])
{
   double total = 0.0;
   for(int i=0; i<FXAI_PAIR_NETWORK_FACTOR_COUNT; i++)
      total += values[i] * values[i];
   if(total <= 0.0)
      return 0.0;
   return MathSqrt(total);
}

double FXAI_PairNetworkCurrencyCosine(const string &lhs_keys[],
                                      const double &lhs_values[],
                                      const int lhs_count,
                                      const string &rhs_keys[],
                                      const double &rhs_values[],
                                      const int rhs_count)
{
   double lhs_norm = FXAI_PairNetworkNorm(lhs_values, lhs_count);
   double rhs_norm = FXAI_PairNetworkNorm(rhs_values, rhs_count);
   if(lhs_norm <= 0.0 || rhs_norm <= 0.0)
      return 0.0;
   return FXAI_Clamp(FXAI_PairNetworkDot(lhs_keys, lhs_values, lhs_count,
                                         rhs_keys, rhs_values, rhs_count) / (lhs_norm * rhs_norm),
                     -1.0,
                     1.0);
}

double FXAI_PairNetworkFactorCosine(const double &lhs[],
                                    const double &rhs[])
{
   double lhs_norm = FXAI_PairNetworkFactorNorm(lhs);
   double rhs_norm = FXAI_PairNetworkFactorNorm(rhs);
   if(lhs_norm <= 0.0 || rhs_norm <= 0.0)
      return 0.0;
   double dot = 0.0;
   for(int i=0; i<FXAI_PAIR_NETWORK_FACTOR_COUNT; i++)
      dot += lhs[i] * rhs[i];
   return FXAI_Clamp(dot / (lhs_norm * rhs_norm), -1.0, 1.0);
}

double FXAI_PairNetworkTopShareCurrency(const double &values[],
                                        const int count)
{
   double total = 0.0;
   double max_abs = 0.0;
   for(int i=0; i<count; i++)
   {
      double abs_value = MathAbs(values[i]);
      total += abs_value;
      if(abs_value > max_abs)
         max_abs = abs_value;
   }
   if(total <= 0.0)
      return 0.0;
   return FXAI_Clamp(max_abs / total, 0.0, 1.0);
}

double FXAI_PairNetworkHerfindahlCurrency(const double &values[],
                                          const int count)
{
   double total = 0.0;
   for(int i=0; i<count; i++)
      total += MathAbs(values[i]);
   if(total <= 0.0)
      return 0.0;
   double score = 0.0;
   for(int i=0; i<count; i++)
   {
      double share = MathAbs(values[i]) / total;
      score += share * share;
   }
   return FXAI_Clamp(score, 0.0, 1.0);
}

double FXAI_PairNetworkTopShareFactor(const double &values[])
{
   double total = 0.0;
   double max_abs = 0.0;
   for(int i=0; i<FXAI_PAIR_NETWORK_FACTOR_COUNT; i++)
   {
      double abs_value = MathAbs(values[i]);
      total += abs_value;
      if(abs_value > max_abs)
         max_abs = abs_value;
   }
   if(total <= 0.0)
      return 0.0;
   return FXAI_Clamp(max_abs / total, 0.0, 1.0);
}

double FXAI_PairNetworkHerfindahlFactor(const double &values[])
{
   double total = 0.0;
   for(int i=0; i<FXAI_PAIR_NETWORK_FACTOR_COUNT; i++)
      total += MathAbs(values[i]);
   if(total <= 0.0)
      return 0.0;
   double score = 0.0;
   for(int i=0; i<FXAI_PAIR_NETWORK_FACTOR_COUNT; i++)
   {
      double share = MathAbs(values[i]) / total;
      score += share * share;
   }
   return FXAI_Clamp(score, 0.0, 1.0);
}

string FXAI_PairNetworkDecisionLabel(const int decision_code)
{
   switch(decision_code)
   {
      case FXAI_PAIR_NETWORK_DECISION_ALLOW_REDUCED: return "ALLOW_REDUCED";
      case FXAI_PAIR_NETWORK_DECISION_SUPPRESS_REDUNDANT: return "SUPPRESS_REDUNDANT";
      case FXAI_PAIR_NETWORK_DECISION_BLOCK_CONTRADICTORY: return "BLOCK_CONTRADICTORY";
      case FXAI_PAIR_NETWORK_DECISION_BLOCK_CONCENTRATION: return "BLOCK_CONCENTRATION";
      case FXAI_PAIR_NETWORK_DECISION_PREFER_ALTERNATIVE: return "PREFER_ALTERNATIVE_EXPRESSION";
   }
   return "ALLOW";
}

void FXAI_PairNetworkBuildSymbolExposure(const FXAIPairNetworkConfig &cfg,
                                         const string symbol,
                                         const int direction,
                                         const double size_units,
                                         string &currency_keys[],
                                         double &currency_values[],
                                         int &currency_count,
                                         double &factor_values[])
{
   string base = "";
   string quote = "";
   FXAI_ParseSymbolLegs(symbol, base, quote);
   if(StringLen(base) != 3 || StringLen(quote) != 3)
      return;
   if(direction != 0 && direction != 1)
      return;
   double signed_units = MathAbs(size_units) * (direction == 1 ? 1.0 : -1.0);
   FXAI_PairNetworkAddKeyValue(currency_keys, currency_values, currency_count, base, signed_units, FXAI_PAIR_NETWORK_MAX_CURRENCIES);
   FXAI_PairNetworkAddKeyValue(currency_keys, currency_values, currency_count, quote, -signed_units, FXAI_PAIR_NETWORK_MAX_CURRENCIES);
   for(int factor_idx=0; factor_idx<FXAI_PAIR_NETWORK_FACTOR_COUNT; factor_idx++)
   {
      factor_values[factor_idx] += signed_units * FXAI_PairNetworkCurrencyFactorWeight(cfg, base, factor_idx);
      factor_values[factor_idx] -= signed_units * FXAI_PairNetworkCurrencyFactorWeight(cfg, quote, factor_idx);
   }
}

double FXAI_PairNetworkStructuralOverlap(const FXAIPairNetworkConfig &cfg,
                                         const string lhs_symbol,
                                         const string rhs_symbol)
{
   string lhs_base = "";
   string lhs_quote = "";
   string rhs_base = "";
   string rhs_quote = "";
   FXAI_ParseSymbolLegs(lhs_symbol, lhs_base, lhs_quote);
   FXAI_ParseSymbolLegs(rhs_symbol, rhs_base, rhs_quote);
   if(StringLen(lhs_base) != 3 || StringLen(lhs_quote) != 3 ||
      StringLen(rhs_base) != 3 || StringLen(rhs_quote) != 3)
      return 0.0;

   double currency_score = 0.0;
   if(lhs_base == rhs_base && lhs_quote == rhs_quote)
      currency_score = 1.0;
   else if(lhs_base == rhs_quote && lhs_quote == rhs_base)
      currency_score = 0.96;
   else if(lhs_base == rhs_base || lhs_quote == rhs_quote)
      currency_score = 0.84;
   else if(lhs_base == rhs_quote || lhs_quote == rhs_base)
      currency_score = 0.72;
   else if(FXAI_SymbolsShareCurrency(lhs_symbol, rhs_symbol))
      currency_score = 0.56;

   string lhs_keys[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   string rhs_keys[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double lhs_values[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double rhs_values[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double lhs_factors[FXAI_PAIR_NETWORK_FACTOR_COUNT];
   double rhs_factors[FXAI_PAIR_NETWORK_FACTOR_COUNT];
   int lhs_count = 0;
   int rhs_count = 0;
   ArrayInitialize(lhs_values, 0.0);
   ArrayInitialize(rhs_values, 0.0);
   ArrayInitialize(lhs_factors, 0.0);
   ArrayInitialize(rhs_factors, 0.0);
   FXAI_PairNetworkBuildSymbolExposure(cfg, lhs_symbol, 1, 1.0, lhs_keys, lhs_values, lhs_count, lhs_factors);
   FXAI_PairNetworkBuildSymbolExposure(cfg, rhs_symbol, 1, 1.0, rhs_keys, rhs_values, rhs_count, rhs_factors);
   for(int i=0; i<FXAI_PAIR_NETWORK_FACTOR_COUNT; i++)
   {
      lhs_factors[i] = MathAbs(lhs_factors[i]);
      rhs_factors[i] = MathAbs(rhs_factors[i]);
   }
   double factor_score = FXAI_Clamp((FXAI_PairNetworkFactorCosine(lhs_factors, rhs_factors) + 1.0) * 0.5, 0.0, 1.0);
   double cluster_bonus = 0.0;
   if(StringFind(lhs_symbol, "USD") >= 0 && StringFind(rhs_symbol, "USD") >= 0)
      cluster_bonus += 0.08;
   if((StringFind(lhs_symbol, "AUD") >= 0 || StringFind(lhs_symbol, "CAD") >= 0 || StringFind(lhs_symbol, "NZD") >= 0 || StringFind(lhs_symbol, "NOK") >= 0) &&
      (StringFind(rhs_symbol, "AUD") >= 0 || StringFind(rhs_symbol, "CAD") >= 0 || StringFind(rhs_symbol, "NZD") >= 0 || StringFind(rhs_symbol, "NOK") >= 0))
      cluster_bonus += 0.12;
   if((StringFind(lhs_symbol, "JPY") >= 0 || StringFind(lhs_symbol, "CHF") >= 0) &&
      (StringFind(rhs_symbol, "JPY") >= 0 || StringFind(rhs_symbol, "CHF") >= 0))
      cluster_bonus += 0.10;
   return FXAI_Clamp(0.64 * currency_score + 0.28 * factor_score + cluster_bonus, 0.0, 1.0);
}

int FXAI_PairNetworkOrderDirection(const ENUM_ORDER_TYPE order_type)
{
   if(order_type == ORDER_TYPE_BUY || order_type == ORDER_TYPE_BUY_LIMIT || order_type == ORDER_TYPE_BUY_STOP || order_type == ORDER_TYPE_BUY_STOP_LIMIT)
      return 1;
   if(order_type == ORDER_TYPE_SELL || order_type == ORDER_TYPE_SELL_LIMIT || order_type == ORDER_TYPE_SELL_STOP || order_type == ORDER_TYPE_SELL_STOP_LIMIT)
      return 0;
   return -1;
}

void FXAI_PairNetworkCollectManagedExposure(const FXAIPairNetworkConfig &cfg,
                                            const string candidate_symbol,
                                            const int candidate_direction,
                                            string &currency_keys[],
                                            double &currency_values[],
                                            int &currency_count,
                                            double &factor_values[],
                                            bool &direct_contradiction)
{
   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      string pos_symbol = PositionGetString(POSITION_SYMBOL);
      int pos_direction = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : 0);
      if(pos_symbol == candidate_symbol && pos_direction != candidate_direction)
         direct_contradiction = true;
      FXAI_PairNetworkBuildSymbolExposure(cfg,
                                          pos_symbol,
                                          pos_direction,
                                          MathMax(PositionGetDouble(POSITION_VOLUME), 0.0),
                                          currency_keys,
                                          currency_values,
                                          currency_count,
                                          factor_values);
   }

   for(int j=OrdersTotal() - 1; j>=0; j--)
   {
      ulong order_ticket = OrderGetTicket(j);
      if(order_ticket == 0) continue;
      if(!OrderSelect(order_ticket)) continue;
      if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
      string order_symbol = OrderGetString(ORDER_SYMBOL);
      int order_direction = FXAI_PairNetworkOrderDirection((ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE));
      if(order_direction < 0)
         continue;
      if(order_symbol == candidate_symbol && order_direction != candidate_direction)
         direct_contradiction = true;
      FXAI_PairNetworkBuildSymbolExposure(cfg,
                                          order_symbol,
                                          order_direction,
                                          MathMax(OrderGetDouble(ORDER_VOLUME_CURRENT), 0.0),
                                          currency_keys,
                                          currency_values,
                                          currency_count,
                                          factor_values);
   }
}

double FXAI_PairNetworkPeerQuality(const FXAIControlPlaneSnapshot &snap)
{
   return FXAI_Clamp(0.26 * FXAI_Clamp(snap.signal_intensity / 4.0, 0.0, 1.0) +
                     0.20 * FXAI_Clamp(snap.policy_enter_prob, 0.0, 1.0) +
                     0.18 * FXAI_Clamp(snap.confidence, 0.0, 1.0) +
                     0.16 * FXAI_Clamp(snap.reliability, 0.0, 1.0) +
                     0.12 * FXAI_Clamp(snap.policy_capital_efficiency, 0.0, 1.0) +
                     0.08 * FXAI_Clamp(snap.policy_portfolio_fit, 0.0, 1.0),
                     0.0,
                     1.0);
}

int FXAI_PairNetworkCollectPeerCandidates(const FXAIPairNetworkConfig &cfg,
                                          const string candidate_symbol,
                                          const int candidate_direction,
                                          string &currency_keys[],
                                          double &currency_values[],
                                          int &currency_count,
                                          double &factor_values[],
                                          bool &direct_contradiction,
                                          FXAIPairNetworkPeerCandidate &peers[])
{
   ArrayResize(peers, 0);
   string found = "";
   long search = FileFindFirst(FXAI_CONTROL_PLANE_DIR + "\\cp_*.tsv", found, FILE_COMMON);
   if(search == INVALID_HANDLE)
      return 0;

   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time <= 0)
      if(!FXAI_MarketDataBarTime(candidate_symbol, PERIOD_M1, 0, now_time))
         now_time = 0;
   long login = (long)AccountInfoInteger(ACCOUNT_LOGIN);
   ulong magic = TradeMagic;
   long self_chart = (long)ChartID();

   bool cont = true;
   while(cont)
   {
      string file_name = FXAI_CONTROL_PLANE_DIR + "\\" + found;
      FXAIControlPlaneSnapshot snap;
      if(FXAI_ReadControlPlaneSnapshotFile(file_name, snap))
      {
         bool stale = (now_time > 0 && snap.bar_time > 0 && (now_time - snap.bar_time) > FXAI_CONTROL_PLANE_TTL_SEC);
         if(stale)
         {
            FileDelete(file_name, FILE_COMMON);
         }
         else if(snap.login == login &&
                 snap.magic == magic &&
                 snap.chart_id != self_chart &&
                 (snap.direction == 0 || snap.direction == 1) &&
                 snap.policy_enter_prob >= 0.05)
         {
            if(snap.symbol == candidate_symbol && snap.direction != candidate_direction)
               direct_contradiction = true;
            double size_units = FXAI_Clamp(0.25 + 0.75 * FXAI_Clamp(snap.policy_size_mult, 0.0, 2.0) *
                                           (0.55 + 0.45 * FXAI_Clamp(snap.policy_enter_prob, 0.0, 1.0)),
                                           0.10,
                                           2.50);
            FXAI_PairNetworkBuildSymbolExposure(cfg,
                                                snap.symbol,
                                                snap.direction,
                                                size_units,
                                                currency_keys,
                                                currency_values,
                                                currency_count,
                                                factor_values);
            int idx = ArraySize(peers);
            if(idx < FXAI_PAIR_NETWORK_MAX_PEERS)
            {
               ArrayResize(peers, idx + 1);
               peers[idx].ready = true;
               peers[idx].symbol = snap.symbol;
               peers[idx].direction = snap.direction;
               peers[idx].size_units = size_units;
               peers[idx].quality = FXAI_PairNetworkPeerQuality(snap);
            }
         }
      }
      cont = FileFindNext(search, found);
   }
   FileFindClose(search);
   return ArraySize(peers);
}

double FXAI_PairNetworkCandidateEdgeScore(void)
{
   double min_move = MathMax(g_ai_last_min_move_points, 0.25);
   double edge_points = g_ai_last_trade_edge_points;
   if(g_prob_calibration_last_ready)
      edge_points = MathMax(g_prob_calibration_last_edge_after_costs, 0.0);
   return FXAI_Clamp(edge_points / (min_move * 2.0), 0.0, 1.0);
}

double FXAI_PairNetworkMacroFit(const double &candidate_factors[],
                                const FXAICrossAssetPairState &cross_state,
                                const FXAIRatesEnginePairState &rates_state)
{
   double score = 0.50;
   if(cross_state.ready)
   {
      if(cross_state.risk_state == "RISK_OFF")
         score += 0.10 * FXAI_Clamp(candidate_factors[2] - candidate_factors[4], -1.0, 1.0);
      else
         score += 0.08 * FXAI_Clamp(candidate_factors[4] - candidate_factors[2], -1.0, 1.0);
      if(cross_state.liquidity_state == "STRESSED")
         score += 0.06 * FXAI_Clamp(candidate_factors[5], -1.0, 1.0);
   }
   if(rates_state.ready)
   {
      if(rates_state.trade_gate == "BLOCK")
         score -= 0.12;
      else if(rates_state.trade_gate == "CAUTION")
         score -= 0.06;
   }
   return FXAI_Clamp(score, 0.0, 1.0);
}

double FXAI_PairNetworkQualityScore(const FXAIPairNetworkConfig &cfg,
                                    const double edge_score,
                                    const double execution_quality_score,
                                    const double calibration_quality,
                                    const double portfolio_fit,
                                    const double macro_fit,
                                    const double overlap_score)
{
   double diversification = FXAI_Clamp(1.0 - overlap_score, 0.0, 1.0);
   return FXAI_Clamp(cfg.weight_edge_after_costs * edge_score +
                     cfg.weight_execution_quality * FXAI_Clamp(execution_quality_score, 0.0, 1.0) +
                     cfg.weight_calibration_quality * FXAI_Clamp(calibration_quality, 0.0, 1.0) +
                     cfg.weight_portfolio_fit * FXAI_Clamp(portfolio_fit, 0.0, 1.0) +
                     cfg.weight_diversification * diversification +
                     cfg.weight_macro_fit * FXAI_Clamp(macro_fit, 0.0, 1.0),
                     0.0,
                     1.0);
}

void FXAI_PairNetworkLoadConfig(FXAIPairNetworkConfig &out)
{
   FXAI_ResetPairNetworkConfig(out);

   int handle = FileOpen(FXAI_PairNetworkConfigFile(),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle != INVALID_HANDLE)
   {
      while(!FileIsEnding(handle))
      {
         string line = FileReadString(handle);
         if(StringLen(line) <= 0)
            continue;
         string parts[];
         int n = StringSplit(line, '\t', parts);
         if(n < 2)
            continue;
         string key = parts[0];
         if(key == "enabled") out.enabled = (StringToInteger(parts[1]) != 0);
         else if(key == "graph_stale_after_sec") out.graph_stale_after_sec = (int)StringToInteger(parts[1]);
         else if(key == "history_points") out.history_points = (int)StringToInteger(parts[1]);
         else if(key == "max_edges_per_pair") out.max_edges_per_pair = (int)StringToInteger(parts[1]);
         else if(key == "fallback_structural_only") out.fallback_structural_only = (StringToInteger(parts[1]) != 0);
         else if(key == "min_empirical_overlap") out.min_empirical_overlap = (int)StringToInteger(parts[1]);
         else if(key == "empirical_lookback_bars") out.empirical_lookback_bars = (int)StringToInteger(parts[1]);
         else if(key == "structural_weight") out.structural_weight = StringToDouble(parts[1]);
         else if(key == "empirical_weight") out.empirical_weight = StringToDouble(parts[1]);
         else if(key == "redundancy_threshold") out.redundancy_threshold = StringToDouble(parts[1]);
         else if(key == "contradiction_threshold") out.contradiction_threshold = StringToDouble(parts[1]);
         else if(key == "concentration_reduce_threshold") out.concentration_reduce_threshold = StringToDouble(parts[1]);
         else if(key == "concentration_block_threshold") out.concentration_block_threshold = StringToDouble(parts[1]);
         else if(key == "execution_overlap_threshold") out.execution_overlap_threshold = StringToDouble(parts[1]);
         else if(key == "reduced_size_multiplier_floor") out.reduced_size_multiplier_floor = StringToDouble(parts[1]);
         else if(key == "preferred_expression_margin") out.preferred_expression_margin = StringToDouble(parts[1]);
         else if(key == "min_incremental_edge_score") out.min_incremental_edge_score = StringToDouble(parts[1]);
         else if(key == "action_mode") out.auto_apply = (parts[1] != "RECOMMEND_ONLY");
         else if(key == "selection_weight" && n >= 3)
         {
            string weight_key = parts[1];
            double weight_value = StringToDouble(parts[2]);
            if(weight_key == "edge_after_costs") out.weight_edge_after_costs = weight_value;
            else if(weight_key == "execution_quality") out.weight_execution_quality = weight_value;
            else if(weight_key == "calibration_quality") out.weight_calibration_quality = weight_value;
            else if(weight_key == "portfolio_fit") out.weight_portfolio_fit = weight_value;
            else if(weight_key == "diversification") out.weight_diversification = weight_value;
            else if(weight_key == "macro_fit") out.weight_macro_fit = weight_value;
         }
         else if(key == "currency_profile" && n >= 4)
         {
            string currency = parts[1];
            string factor_name = parts[2];
            double factor_value = StringToDouble(parts[3]);
            int idx = FXAI_PairNetworkFindCurrencyProfile(out, currency);
            if(idx < 0 && out.currency_profile_count < FXAI_PAIR_NETWORK_MAX_CURRENCY_PROFILES)
            {
               idx = out.currency_profile_count;
               FXAI_ResetPairNetworkCurrencyProfile(out.currency_profiles[idx]);
               out.currency_profiles[idx].currency = currency;
               out.currency_profile_count++;
            }
            if(idx >= 0)
            {
               if(factor_name == "usd_bloc") out.currency_profiles[idx].usd_bloc = factor_value;
               else if(factor_name == "eur_rates") out.currency_profiles[idx].eur_rates = factor_value;
               else if(factor_name == "safe_haven") out.currency_profiles[idx].safe_haven = factor_value;
               else if(factor_name == "commodity_fx") out.currency_profiles[idx].commodity_fx = factor_value;
               else if(factor_name == "risk_on") out.currency_profiles[idx].risk_on = factor_value;
               else if(factor_name == "liquidity_stress") out.currency_profiles[idx].liquidity_stress = factor_value;
               else if(factor_name == "macro_shock") out.currency_profiles[idx].macro_shock = factor_value;
            }
         }
      }
      FileClose(handle);
   }

   int status = FileOpen(FXAI_PairNetworkStatusFile(),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(status != INVALID_HANDLE)
   {
      while(!FileIsEnding(status))
      {
         string line = FileReadString(status);
         if(StringLen(line) <= 0)
            continue;
         string parts[];
         int n = StringSplit(line, '\t', parts);
         if(n < 2)
            continue;
         string key = parts[0];
         string value = parts[1];
         if(key == "graph_mode") out.graph_mode = value;
         else if(key == "fallback_graph_used") out.fallback_graph_used = (StringToInteger(value) != 0);
         else if(key == "partial_dependency_data") out.partial_dependency_data = (StringToInteger(value) != 0);
         else if(key == "graph_stale") out.graph_stale = (StringToInteger(value) != 0);
      }
      FileClose(status);
   }

   out.ready = true;
}

void FXAI_PairNetworkEnsureConfigLoaded(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time > 0 &&
      g_pair_network_cfg_cache_loaded_at > 0 &&
      (now_time - g_pair_network_cfg_cache_loaded_at) < 60)
      return;
   FXAI_PairNetworkLoadConfig(g_pair_network_cfg_cache);
   g_pair_network_cfg_cache_loaded_at = now_time;
}

string FXAI_PairNetworkCurrencyExposureCSV(const string &keys[],
                                           const double &values[],
                                           const int count)
{
   string out = "";
   for(int i=0; i<count; i++)
   {
      if(MathAbs(values[i]) <= 1e-9)
         continue;
      if(StringLen(out) > 0)
         out += "; ";
      out += keys[i] + ":" + DoubleToString(values[i], 4);
   }
   return out;
}

string FXAI_PairNetworkFactorExposureCSV(const double &values[])
{
   string out = "";
   for(int i=0; i<FXAI_PAIR_NETWORK_FACTOR_COUNT; i++)
   {
      if(MathAbs(values[i]) <= 1e-9)
         continue;
      if(StringLen(out) > 0)
         out += "; ";
      out += FXAI_PairNetworkFactorName(i) + ":" + DoubleToString(values[i], 4);
   }
   return out;
}

bool FXAI_PairNetworkEvaluate(const string symbol,
                              const int direction,
                              const FXAINewsPulsePairState &news_state,
                              const FXAIRatesEnginePairState &rates_state,
                              const FXAICrossAssetPairState &cross_state,
                              const FXAIMicrostructurePairState &micro_state,
                              const FXAIExecutionQualityPairState &execution_state,
                              FXAIPairNetworkDecisionState &out)
{
   FXAI_ResetPairNetworkDecisionState(out);
   FXAI_PairNetworkEnsureConfigLoaded();
   FXAIPairNetworkConfig cfg = g_pair_network_cfg_cache;
   if(!PairNetworkEnabled || !cfg.ready || !cfg.enabled || (direction != 0 && direction != 1))
      return false;

   string portfolio_currency_keys[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double portfolio_currency_values[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double portfolio_factor_values[FXAI_PAIR_NETWORK_FACTOR_COUNT];
   ArrayInitialize(portfolio_currency_values, 0.0);
   ArrayInitialize(portfolio_factor_values, 0.0);
   int portfolio_currency_count = 0;
   bool direct_contradiction = false;

   FXAI_PairNetworkCollectManagedExposure(cfg,
                                          symbol,
                                          direction,
                                          portfolio_currency_keys,
                                          portfolio_currency_values,
                                          portfolio_currency_count,
                                          portfolio_factor_values,
                                          direct_contradiction);

   FXAIPairNetworkPeerCandidate peers[];
   FXAI_PairNetworkCollectPeerCandidates(cfg,
                                         symbol,
                                         direction,
                                         portfolio_currency_keys,
                                         portfolio_currency_values,
                                         portfolio_currency_count,
                                         portfolio_factor_values,
                                         direct_contradiction,
                                         peers);

   string candidate_currency_keys[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double candidate_currency_values[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double candidate_factor_values[FXAI_PAIR_NETWORK_FACTOR_COUNT];
   ArrayInitialize(candidate_currency_values, 0.0);
   ArrayInitialize(candidate_factor_values, 0.0);
   int candidate_currency_count = 0;
   FXAI_PairNetworkBuildSymbolExposure(cfg,
                                       symbol,
                                       direction,
                                       1.0,
                                       candidate_currency_keys,
                                       candidate_currency_values,
                                       candidate_currency_count,
                                       candidate_factor_values);

   string after_currency_keys[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double after_currency_values[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
   double after_factor_values[FXAI_PAIR_NETWORK_FACTOR_COUNT];
   int after_currency_count = 0;
   ArrayInitialize(after_currency_values, 0.0);
   ArrayInitialize(after_factor_values, 0.0);
   for(int i=0; i<portfolio_currency_count; i++)
   {
      after_currency_keys[i] = portfolio_currency_keys[i];
      after_currency_values[i] = portfolio_currency_values[i];
      after_currency_count++;
   }
   for(int j=0; j<FXAI_PAIR_NETWORK_FACTOR_COUNT; j++)
      after_factor_values[j] = portfolio_factor_values[j];
   FXAI_PairNetworkBuildSymbolExposure(cfg,
                                       symbol,
                                       direction,
                                       1.0,
                                       after_currency_keys,
                                       after_currency_values,
                                       after_currency_count,
                                       after_factor_values);

   double currency_alignment = FXAI_PairNetworkCurrencyCosine(candidate_currency_keys,
                                                              candidate_currency_values,
                                                              candidate_currency_count,
                                                              portfolio_currency_keys,
                                                              portfolio_currency_values,
                                                              portfolio_currency_count);
   double factor_alignment = FXAI_PairNetworkFactorCosine(candidate_factor_values,
                                                          portfolio_factor_values);
   double overlap_score = FXAI_Clamp(0.58 * MathMax(currency_alignment, 0.0) +
                                     0.42 * MathMax(factor_alignment, 0.0),
                                     0.0,
                                     1.0);
   double contradiction_core = FXAI_Clamp(0.64 * MathMax(-currency_alignment, 0.0) +
                                          0.36 * MathMax(-factor_alignment, 0.0),
                                          0.0,
                                          1.0);
   double contradiction_score = (direct_contradiction ? 1.0 : contradiction_core);

   double before_currency_concentration = MathMax(FXAI_PairNetworkTopShareCurrency(portfolio_currency_values, portfolio_currency_count),
                                                  FXAI_PairNetworkHerfindahlCurrency(portfolio_currency_values, portfolio_currency_count));
   double before_factor_concentration = MathMax(FXAI_PairNetworkTopShareFactor(portfolio_factor_values),
                                                FXAI_PairNetworkHerfindahlFactor(portfolio_factor_values));
   double after_currency_concentration = MathMax(FXAI_PairNetworkTopShareCurrency(after_currency_values, after_currency_count),
                                                 FXAI_PairNetworkHerfindahlCurrency(after_currency_values, after_currency_count));
   double after_factor_concentration = MathMax(FXAI_PairNetworkTopShareFactor(after_factor_values),
                                               FXAI_PairNetworkHerfindahlFactor(after_factor_values));
   double concentration_score = FXAI_Clamp(0.54 * MathMax(after_currency_concentration, after_factor_concentration) +
                                           0.24 * MathMax(after_currency_concentration - before_currency_concentration, 0.0) +
                                           0.22 * MathMax(after_factor_concentration - before_factor_concentration, 0.0),
                                           0.0,
                                           1.0);

   double news_risk_score = (NewsPulseEnabled && news_state.ready ? news_state.news_risk_score : 0.0);
   double execution_stress_score = 0.0;
   if(ExecutionQualityEnabled && execution_state.ready)
      execution_stress_score = FXAI_Clamp(MathMax(MathMax(execution_state.spread_widening_risk, execution_state.slippage_risk),
                                                  1.0 - execution_state.execution_quality_score),
                                          0.0,
                                          1.0);
   else if(MicrostructureEnabled && micro_state.ready)
      execution_stress_score = FXAI_Clamp(MathMax(micro_state.liquidity_stress_score, micro_state.hostile_execution_score), 0.0, 1.0);

   double edge_score = FXAI_PairNetworkCandidateEdgeScore();
   double execution_quality_score = (ExecutionQualityEnabled && execution_state.ready
                                     ? FXAI_Clamp(execution_state.execution_quality_score, 0.0, 1.0)
                                     : FXAI_Clamp(1.0 - execution_stress_score, 0.0, 1.0));
   double calibration_quality = (g_prob_calibration_last_ready
                                 ? FXAI_Clamp(g_prob_calibration_last_quality, 0.0, 1.0)
                                 : FXAI_Clamp(g_ai_last_reliability, 0.0, 1.0));
   double portfolio_fit = FXAI_Clamp(g_policy_last_portfolio_fit, 0.0, 1.0);
   double macro_fit = FXAI_PairNetworkMacroFit(candidate_factor_values, cross_state, rates_state);
   double candidate_overlap = FXAI_Clamp(MathMax(overlap_score, concentration_score), 0.0, 1.0);
   double candidate_quality = FXAI_PairNetworkQualityScore(cfg,
                                                           edge_score,
                                                           execution_quality_score,
                                                           calibration_quality,
                                                           portfolio_fit,
                                                           macro_fit,
                                                           candidate_overlap);

   string preferred_expression = "";
   double preferred_quality = candidate_quality;
   for(int peer_idx=0; peer_idx<ArraySize(peers); peer_idx++)
   {
      string peer_currency_keys[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
      double peer_currency_values[FXAI_PAIR_NETWORK_MAX_CURRENCIES];
      double peer_factor_values[FXAI_PAIR_NETWORK_FACTOR_COUNT];
      int peer_currency_count = 0;
      ArrayInitialize(peer_currency_values, 0.0);
      ArrayInitialize(peer_factor_values, 0.0);
      FXAI_PairNetworkBuildSymbolExposure(cfg,
                                          peers[peer_idx].symbol,
                                          peers[peer_idx].direction,
                                          peers[peer_idx].size_units,
                                          peer_currency_keys,
                                          peer_currency_values,
                                          peer_currency_count,
                                          peer_factor_values);
      double same_view = FXAI_Clamp(0.62 * MathMax(FXAI_PairNetworkCurrencyCosine(candidate_currency_keys,
                                                                                   candidate_currency_values,
                                                                                   candidate_currency_count,
                                                                                   peer_currency_keys,
                                                                                   peer_currency_values,
                                                                                   peer_currency_count), 0.0) +
                                    0.38 * MathMax(FXAI_PairNetworkFactorCosine(candidate_factor_values,
                                                                                peer_factor_values), 0.0),
                                    0.0,
                                    1.0);
      same_view = MathMax(same_view, FXAI_PairNetworkStructuralOverlap(cfg, symbol, peers[peer_idx].symbol));
      if(same_view < 0.60)
         continue;
      if(peers[peer_idx].quality > preferred_quality + cfg.preferred_expression_margin)
      {
         preferred_quality = peers[peer_idx].quality;
         preferred_expression = peers[peer_idx].symbol;
      }
   }

   double redundancy_score = FXAI_Clamp(0.56 * overlap_score +
                                        0.18 * news_risk_score * overlap_score +
                                        0.16 * execution_stress_score * overlap_score +
                                        0.10 * (StringLen(preferred_expression) > 0 ? 1.0 : 0.0),
                                        0.0,
                                        1.0);
   double execution_overlap_score = FXAI_Clamp(execution_stress_score * overlap_score, 0.0, 1.0);
   double conflict_score = MathMax(contradiction_score, MathMax(redundancy_score, concentration_score));

   int decision_code = FXAI_PAIR_NETWORK_DECISION_ALLOW;
   double size_multiplier = 1.0;
   if(direct_contradiction)
      FXAI_PairNetworkAppendReason(out, "DIRECT_SYMBOL_CONTRADICTION");
   if(contradiction_score >= cfg.contradiction_threshold)
   {
      decision_code = FXAI_PAIR_NETWORK_DECISION_BLOCK_CONTRADICTORY;
      size_multiplier = 0.0;
      FXAI_PairNetworkAppendReason(out, "CURRENCY_EXPOSURE_CONFLICT");
   }
   else if(StringLen(preferred_expression) > 0 && MathMax(overlap_score, redundancy_score) >= 0.55)
   {
      decision_code = FXAI_PAIR_NETWORK_DECISION_PREFER_ALTERNATIVE;
      size_multiplier = 0.0;
      FXAI_PairNetworkAppendReason(out, "BETTER_ALTERNATIVE_EXPRESSION");
   }
   else if(concentration_score >= cfg.concentration_block_threshold && candidate_quality < 0.72)
   {
      decision_code = FXAI_PAIR_NETWORK_DECISION_BLOCK_CONCENTRATION;
      size_multiplier = 0.0;
      FXAI_PairNetworkAppendReason(out, "HIDDEN_CURRENCY_CONCENTRATION");
   }
   else if(redundancy_score >= cfg.redundancy_threshold && candidate_quality < cfg.min_incremental_edge_score)
   {
      decision_code = FXAI_PAIR_NETWORK_DECISION_SUPPRESS_REDUNDANT;
      size_multiplier = 0.0;
      FXAI_PairNetworkAppendReason(out, "LOW_INCREMENTAL_PORTFOLIO_EDGE");
   }
   else if(concentration_score >= cfg.concentration_reduce_threshold ||
           redundancy_score >= cfg.redundancy_threshold ||
           execution_overlap_score >= cfg.execution_overlap_threshold)
   {
      decision_code = FXAI_PAIR_NETWORK_DECISION_ALLOW_REDUCED;
      size_multiplier = FXAI_Clamp(1.0 - 0.55 * MathMax(redundancy_score, MathMax(concentration_score, execution_overlap_score)),
                                   cfg.reduced_size_multiplier_floor,
                                   0.95);
   }

   int dominant_currency_idx = -1;
   double dominant_currency_abs = 0.0;
   for(int k=0; k<candidate_currency_count; k++)
   {
      double abs_value = MathAbs(candidate_currency_values[k]);
      if(abs_value > dominant_currency_abs)
      {
         dominant_currency_abs = abs_value;
         dominant_currency_idx = k;
      }
   }
   if(redundancy_score >= cfg.redundancy_threshold && dominant_currency_idx >= 0)
   {
      string dominant_direction = (candidate_currency_values[dominant_currency_idx] >= 0.0 ? "LONG" : "SHORT");
      FXAI_PairNetworkAppendReason(out, "DUPLICATES_EXISTING_" + candidate_currency_keys[dominant_currency_idx] + "_" + dominant_direction + "_EXPOSURE");
   }
   if(MathAbs(candidate_factor_values[3]) >= 0.35 && redundancy_score >= 0.52)
      FXAI_PairNetworkAppendReason(out, "HIGH_COMMODITY_BLOC_OVERLAP");
   if(concentration_score >= cfg.concentration_reduce_threshold)
      FXAI_PairNetworkAppendReason(out, "FACTOR_CONCENTRATION_ELEVATED");
   if(execution_overlap_score >= cfg.execution_overlap_threshold)
      FXAI_PairNetworkAppendReason(out, "EXECUTION_STRESS_OVERLAP");
   if(news_risk_score >= 0.55 && MathMax(redundancy_score, concentration_score) >= 0.45)
      FXAI_PairNetworkAppendReason(out, "EVENT_STACKING_RISK");
   if(macro_fit < 0.42)
      FXAI_PairNetworkAppendReason(out, "INCONSISTENT_MACRO_EXPRESSION");
   if(StringLen(preferred_expression) > 0)
      FXAI_PairNetworkAppendReason(out, "PREFERRED_EXPRESSION_" + preferred_expression);

   out.ready = true;
   out.fallback_graph_used = cfg.fallback_graph_used;
   out.partial_dependency_data = cfg.partial_dependency_data;
   out.graph_stale = cfg.graph_stale;
   out.generated_at = TimeCurrent();
   if(out.generated_at <= 0)
      out.generated_at = TimeTradeServer();
   out.symbol = symbol;
   out.direction = direction;
   out.decision = FXAI_PairNetworkDecisionLabel(decision_code);
   out.conflict_score = FXAI_Clamp(conflict_score, 0.0, 1.0);
   out.redundancy_score = FXAI_Clamp(redundancy_score, 0.0, 1.0);
   out.contradiction_score = FXAI_Clamp(contradiction_score, 0.0, 1.0);
   out.concentration_score = FXAI_Clamp(concentration_score, 0.0, 1.0);
   out.currency_concentration = FXAI_Clamp(after_currency_concentration, 0.0, 1.0);
   out.factor_concentration = FXAI_Clamp(after_factor_concentration, 0.0, 1.0);
   out.recommended_size_multiplier = FXAI_Clamp(size_multiplier, 0.0, 1.0);
   out.preferred_expression = preferred_expression;
   out.currency_exposure_csv = FXAI_PairNetworkCurrencyExposureCSV(after_currency_keys, after_currency_values, after_currency_count);
   out.factor_exposure_csv = FXAI_PairNetworkFactorExposureCSV(after_factor_values);

   g_pair_network_last_ready = out.ready;
   g_pair_network_last_fallback_graph_used = out.fallback_graph_used;
   g_pair_network_last_partial_dependency_data = out.partial_dependency_data;
   g_pair_network_last_graph_stale = out.graph_stale;
   g_pair_network_last_generated_at = out.generated_at;
   g_pair_network_last_symbol = symbol;
   g_pair_network_last_decision = out.decision;
   g_pair_network_last_conflict_score = out.conflict_score;
   g_pair_network_last_redundancy_score = out.redundancy_score;
   g_pair_network_last_contradiction_score = out.contradiction_score;
   g_pair_network_last_concentration_score = out.concentration_score;
   g_pair_network_last_currency_concentration = out.currency_concentration;
   g_pair_network_last_factor_concentration = out.factor_concentration;
   g_pair_network_last_recommended_size_multiplier = out.recommended_size_multiplier;
   g_pair_network_last_preferred_expression = out.preferred_expression;
   g_pair_network_last_reasons_csv = FXAI_PairNetworkReasonsCSV(out);

   return true;
}

void FXAI_PairNetworkWriteRuntimeArtifacts(const string symbol,
                                           const FXAIPairNetworkDecisionState &state)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate("FXAI\\Runtime", FILE_COMMON);

   int handle = FileOpen(FXAI_PairNetworkRuntimeStateFile(symbol),
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle != INVALID_HANDLE)
   {
      FileWriteString(handle, "symbol\t" + symbol + "\r\n");
      FileWriteString(handle, "generated_at\t" + IntegerToString((int)state.generated_at) + "\r\n");
      FileWriteString(handle, "decision\t" + state.decision + "\r\n");
      FileWriteString(handle, "fallback_graph_used\t" + IntegerToString(state.fallback_graph_used ? 1 : 0) + "\r\n");
      FileWriteString(handle, "partial_dependency_data\t" + IntegerToString(state.partial_dependency_data ? 1 : 0) + "\r\n");
      FileWriteString(handle, "graph_stale\t" + IntegerToString(state.graph_stale ? 1 : 0) + "\r\n");
      FileWriteString(handle, "conflict_score\t" + DoubleToString(state.conflict_score, 6) + "\r\n");
      FileWriteString(handle, "redundancy_score\t" + DoubleToString(state.redundancy_score, 6) + "\r\n");
      FileWriteString(handle, "contradiction_score\t" + DoubleToString(state.contradiction_score, 6) + "\r\n");
      FileWriteString(handle, "concentration_score\t" + DoubleToString(state.concentration_score, 6) + "\r\n");
      FileWriteString(handle, "currency_concentration\t" + DoubleToString(state.currency_concentration, 6) + "\r\n");
      FileWriteString(handle, "factor_concentration\t" + DoubleToString(state.factor_concentration, 6) + "\r\n");
      FileWriteString(handle, "recommended_size_multiplier\t" + DoubleToString(state.recommended_size_multiplier, 6) + "\r\n");
      FileWriteString(handle, "preferred_expression\t" + state.preferred_expression + "\r\n");
      FileWriteString(handle, "currency_exposure_csv\t" + state.currency_exposure_csv + "\r\n");
      FileWriteString(handle, "factor_exposure_csv\t" + state.factor_exposure_csv + "\r\n");
      FileWriteString(handle, "reasons_csv\t" + FXAI_PairNetworkReasonsCSV(state) + "\r\n");
      FileClose(handle);
   }

   int hist = FileOpen(FXAI_PairNetworkRuntimeHistoryFile(symbol),
                       FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                       FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(hist == INVALID_HANDLE)
      hist = FileOpen(FXAI_PairNetworkRuntimeHistoryFile(symbol),
                      FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                      FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(hist != INVALID_HANDLE)
   {
      FileSeek(hist, 0, SEEK_END);
      string json = "{";
      json += "\"generated_at\":\"" + FXAI_PairNetworkISO8601(state.generated_at) + "\",";
      json += "\"symbol\":\"" + FXAI_PairNetworkJSONEscape(symbol) + "\",";
      json += "\"decision\":\"" + FXAI_PairNetworkJSONEscape(state.decision) + "\",";
      json += "\"fallback_graph_used\":" + IntegerToString(state.fallback_graph_used ? 1 : 0) + ",";
      json += "\"partial_dependency_data\":" + IntegerToString(state.partial_dependency_data ? 1 : 0) + ",";
      json += "\"graph_stale\":" + IntegerToString(state.graph_stale ? 1 : 0) + ",";
      json += "\"conflict_score\":" + DoubleToString(state.conflict_score, 6) + ",";
      json += "\"redundancy_score\":" + DoubleToString(state.redundancy_score, 6) + ",";
      json += "\"contradiction_score\":" + DoubleToString(state.contradiction_score, 6) + ",";
      json += "\"concentration_score\":" + DoubleToString(state.concentration_score, 6) + ",";
      json += "\"currency_concentration\":" + DoubleToString(state.currency_concentration, 6) + ",";
      json += "\"factor_concentration\":" + DoubleToString(state.factor_concentration, 6) + ",";
      json += "\"recommended_size_multiplier\":" + DoubleToString(state.recommended_size_multiplier, 6) + ",";
      json += "\"preferred_expression\":\"" + FXAI_PairNetworkJSONEscape(state.preferred_expression) + "\",";
      json += "\"currency_exposure_csv\":\"" + FXAI_PairNetworkJSONEscape(state.currency_exposure_csv) + "\",";
      json += "\"factor_exposure_csv\":\"" + FXAI_PairNetworkJSONEscape(state.factor_exposure_csv) + "\",";
      json += "\"reason_codes\":[";
      for(int i=0; i<state.reason_count; i++)
      {
         if(i > 0)
            json += ",";
         json += "\"" + FXAI_PairNetworkJSONEscape(state.reasons[i]) + "\"";
      }
      json += "]}";
      FileWriteString(hist, json + "\r\n");
      FileClose(hist);
   }
}

#endif // __FXAI_RUNTIME_TRADE_PAIR_NETWORK_MQH__
