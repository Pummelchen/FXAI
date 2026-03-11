#ifndef __FXAI_AUDIT_CORE_MQH__
#define __FXAI_AUDIT_CORE_MQH__

#include "..\Engine\core.mqh"
#include "..\Engine\data_pipeline.mqh"
#include "..\API\api.mqh"

#define FXAI_AUDIT_REPORT_DIR "FXAI\\Audit"
#define FXAI_AUDIT_REPORT_FILE "FXAI\\Audit\\fxai_audit_report.tsv"
#define FXAI_AUDIT_ISSUE_INVALID_PRED     1
#define FXAI_AUDIT_ISSUE_OVERTRADES_NOISE 2
#define FXAI_AUDIT_ISSUE_MISSES_TREND     4
#define FXAI_AUDIT_ISSUE_CALIBRATION_DRIFT 8
#define FXAI_AUDIT_ISSUE_RESET_DRIFT      16
#define FXAI_AUDIT_ISSUE_SEQUENCE_WEAK    32
#define FXAI_AUDIT_ISSUE_DEAD_OUTPUT      64
#define FXAI_AUDIT_ISSUE_SIDE_COLLAPSE    128

struct FXAIAuditScenarioSpec
{
   int id;
   string name;
   double drift_per_bar;
   double sigma_per_bar;
   double mean_revert_strength;
   double vol_cluster;
   double spike_prob;
   double spike_scale;
   double spread_points;
};

struct FXAIAuditScenarioMetrics
{
   int ai_id;
   string ai_name;
   int family;
   string scenario;
   int bars_total;
   int samples_total;
   int valid_preds;
   int invalid_preds;
   int buy_count;
   int sell_count;
   int skip_count;
   int true_buy_count;
   int true_sell_count;
   int true_skip_count;
   int exact_match_count;
   int directional_eval_count;
   int directional_correct_count;
   double trend_alignment_sum;
   int trend_alignment_count;
   double conf_sum;
   double rel_sum;
   double move_sum;
   double dir_conf_sum;
   double dir_hit_sum;
   double skip_ratio;
   double active_ratio;
   double bias_abs;
   double conf_drift;
   double reset_delta;
   double sequence_delta;
   double score;
   int issue_flags;
};

class CFXAIAuditRng
{
private:
   ulong m_state;

public:
   void Seed(const ulong seed)
   {
      m_state = (seed == 0 ? (ulong)881726454 : seed);
   }

   ulong NextU64(void)
   {
      m_state ^= (m_state << 13);
      m_state ^= (m_state >> 7);
      m_state ^= (m_state << 17);
      return m_state;
   }

   double NextUnit(void)
   {
      ulong lo31 = (ulong)(NextU64() % 2147483647);
      return (double)lo31 / 2147483646.0;
   }

   double NextSigned(void)
   {
      return 2.0 * NextUnit() - 1.0;
   }

   double NextNormal(void)
   {
      double u1 = NextUnit();
      double u2 = NextUnit();
      if(u1 < 1e-9) u1 = 1e-9;
      return MathSqrt(-2.0 * MathLog(u1)) * MathCos(2.0 * M_PI * u2);
   }
};

double FXAI_AuditGetArrayValue(const double &arr[],
                               const int idx,
                               const double def_value)
{
   if(idx >= 0 && idx < ArraySize(arr)) return arr[idx];
   return def_value;
}

double FXAI_AuditGetIntArrayMean(const int &arr[],
                                 const int start_idx,
                                 const int width,
                                 const double fallback)
{
   int n = ArraySize(arr);
   if(n <= 0 || start_idx < 0 || start_idx >= n || width <= 0)
      return MathMax(fallback, 0.10);

   int end = start_idx + width;
   if(end > n) end = n;
   double sum = 0.0;
   int used = 0;
   for(int i=start_idx; i<end; i++)
   {
      double v = (double)arr[i];
      if(v <= 0.0) continue;
      sum += v;
      used++;
   }
   if(used <= 0) return MathMax(fallback, 0.10);
   return sum / (double)used;
}

int FXAI_AuditClampHorizon(const int h_in)
{
   int h = h_in;
   if(h < 1) h = 1;
   if(h > 720) h = 720;
   return h;
}

int FXAI_AuditGetHorizonSlot(const int horizon_minutes)
{
   int h = FXAI_AuditClampHorizon(horizon_minutes);
   if(h <= 3) return 0;
   if(h <= 5) return 1;
   if(h <= 8) return 2;
   if(h <= 13) return 3;
   if(h <= 21) return 4;
   if(h <= 34) return 5;
   if(h <= 55) return 6;
   return 7;
}

int FXAI_AuditGetStaticRegimeId(const datetime sample_time,
                                const double spread_points,
                                const double spread_ref,
                                const double vol_proxy_abs,
                                const double vol_ref)
{
   MqlDateTime dt;
   TimeToStruct(sample_time > 0 ? sample_time : TimeCurrent(), dt);
   int hour = dt.hour;
   if(hour < 0) hour = 0;
   if(hour > 23) hour = 23;

   int sess = 0;
   if(hour < 8) sess = 0;
   else if(hour < 16) sess = 1;
   else sess = 2;

   double sp_ref = MathMax(spread_ref, 0.10);
   double vp_ref = MathMax(MathAbs(vol_ref), 1e-6);
   int spread_hi = (spread_points > (1.15 * sp_ref + 0.10) ? 1 : 0);
   int vol_hi = (MathAbs(vol_proxy_abs) > (1.15 * vp_ref + 0.02) ? 1 : 0);

   int regime = sess * 4 + vol_hi * 2 + spread_hi;
   if(regime < 0) regime = 0;
   if(regime >= FXAI_PLUGIN_REGIME_BUCKETS) regime = FXAI_PLUGIN_REGIME_BUCKETS - 1;
   return regime;
}

void FXAI_AuditDefaultHyperParams(const int ai_idx,
                                  FXAIAIHyperParams &hp)
{
   hp.lr = 0.0100;
   hp.l2 = 0.0030;
   hp.ftrl_alpha = 0.08;
   hp.ftrl_beta = 1.0;
   hp.ftrl_l1 = 0.0005;
   hp.ftrl_l2 = 0.0100;
   hp.pa_c = 4.0;
   hp.pa_margin = 1.2;
   hp.xgb_lr = 0.08;
   hp.xgb_l2 = 0.02;
   hp.xgb_split = 0.50;
   hp.mlp_lr = 0.0100;
   hp.mlp_l2 = 0.0030;
   hp.mlp_init = 0.10;
   hp.quantile_lr = 0.0100;
   hp.quantile_l2 = 0.0030;
   hp.enhash_lr = 0.0100;
   hp.enhash_l1 = 0.0000;
   hp.enhash_l2 = 0.0050;
   hp.tcn_layers = 4.0;
   hp.tcn_kernel = 3.0;
   hp.tcn_dilation_base = 2.0;

   switch(ai_idx)
   {
      case (int)AI_M1SYNC:
      case (int)AI_BUY_ONLY:
      case (int)AI_SELL_ONLY:
      case (int)AI_RANDOM_NOSKIP:
         hp.lr = 0.0;
         hp.l2 = 0.0;
         break;
      case (int)AI_FTRL_LOGIT:
         hp.ftrl_alpha = 0.08;
         hp.ftrl_beta = 1.0;
         hp.ftrl_l1 = 0.0000;
         hp.ftrl_l2 = 0.01;
         break;
      case (int)AI_PA_LINEAR:
         hp.lr = 0.06;
         hp.l2 = 0.003;
         hp.pa_c = 4.0;
         hp.pa_margin = 1.2;
         break;
      case (int)AI_TCN:
         hp.lr = 0.006;
         hp.l2 = 0.002;
         hp.tcn_layers = 4.0;
         hp.tcn_kernel = 3.0;
         hp.tcn_dilation_base = 2.0;
         break;
      case (int)AI_LSTM:
      case (int)AI_LSTMG:
      case (int)AI_TFT:
      case (int)AI_AUTOFORMER:
      case (int)AI_PATCHTST:
      case (int)AI_CHRONOS:
      case (int)AI_TIMESFM:
      case (int)AI_TST:
      case (int)AI_STMN:
      case (int)AI_S4:
      case (int)AI_GEODESICATTENTION:
         hp.lr = 0.006;
         hp.l2 = 0.002;
         break;
      default:
         break;
   }
}

void FXAI_AuditFillScenarioSpec(const int scenario_id,
                                FXAIAuditScenarioSpec &spec)
{
   spec.id = scenario_id;
   spec.name = "random_walk";
   spec.drift_per_bar = 0.0;
   spec.sigma_per_bar = 0.00018;
   spec.mean_revert_strength = 0.0;
   spec.vol_cluster = 0.0;
   spec.spike_prob = 0.0;
   spec.spike_scale = 0.0;
   spec.spread_points = 1.2;

   switch(scenario_id)
   {
      case 1:
         spec.name = "drift_up";
         spec.drift_per_bar = 0.00010;
         spec.sigma_per_bar = 0.00015;
         spec.spread_points = 1.0;
         break;
      case 2:
         spec.name = "drift_down";
         spec.drift_per_bar = -0.00010;
         spec.sigma_per_bar = 0.00015;
         spec.spread_points = 1.0;
         break;
      case 3:
         spec.name = "mean_revert";
         spec.drift_per_bar = 0.0;
         spec.sigma_per_bar = 0.00018;
         spec.mean_revert_strength = 0.22;
         spec.spread_points = 1.3;
         break;
      case 4:
         spec.name = "vol_cluster";
         spec.drift_per_bar = 0.0;
         spec.sigma_per_bar = 0.00018;
         spec.vol_cluster = 0.85;
         spec.spike_prob = 0.01;
         spec.spike_scale = 4.0;
         spec.spread_points = 1.8;
         break;
      case 5:
         spec.name = "monotonic_up";
         spec.drift_per_bar = 0.00022;
         spec.sigma_per_bar = 0.00003;
         spec.spread_points = 0.8;
         break;
      case 6:
         spec.name = "monotonic_down";
         spec.drift_per_bar = -0.00022;
         spec.sigma_per_bar = 0.00003;
         spec.spread_points = 0.8;
         break;
      case 7:
         spec.name = "regime_shift";
         spec.drift_per_bar = 0.00008;
         spec.sigma_per_bar = 0.00015;
         spec.vol_cluster = 0.55;
         spec.spike_prob = 0.005;
         spec.spike_scale = 3.0;
         spec.spread_points = 1.5;
         break;
      default:
         break;
   }
}

void FXAI_AuditResetMetrics(FXAIAuditScenarioMetrics &m,
                            const int ai_id,
                            const string ai_name,
                            const int family,
                            const string scenario,
                            const int bars_total)
{
   m.ai_id = ai_id;
   m.ai_name = ai_name;
   m.family = family;
   m.scenario = scenario;
   m.bars_total = bars_total;
   m.samples_total = 0;
   m.valid_preds = 0;
   m.invalid_preds = 0;
   m.buy_count = 0;
   m.sell_count = 0;
   m.skip_count = 0;
   m.true_buy_count = 0;
   m.true_sell_count = 0;
   m.true_skip_count = 0;
   m.exact_match_count = 0;
   m.directional_eval_count = 0;
   m.directional_correct_count = 0;
   m.trend_alignment_sum = 0.0;
   m.trend_alignment_count = 0;
   m.conf_sum = 0.0;
   m.rel_sum = 0.0;
   m.move_sum = 0.0;
   m.dir_conf_sum = 0.0;
   m.dir_hit_sum = 0.0;
   m.skip_ratio = 0.0;
   m.active_ratio = 0.0;
   m.bias_abs = 0.0;
   m.conf_drift = 0.0;
   m.reset_delta = 0.0;
   m.sequence_delta = 0.0;
   m.score = 0.0;
   m.issue_flags = 0;
}

int FXAI_AuditDecisionFromPred(const FXAIAIPredictionV4 &pred)
{
   int best = (int)FXAI_LABEL_SKIP;
   double best_p = pred.class_probs[(int)FXAI_LABEL_SKIP];
   if(pred.class_probs[(int)FXAI_LABEL_BUY] > best_p)
   {
      best = (int)FXAI_LABEL_BUY;
      best_p = pred.class_probs[(int)FXAI_LABEL_BUY];
   }
   if(pred.class_probs[(int)FXAI_LABEL_SELL] > best_p)
      best = (int)FXAI_LABEL_SELL;
   return best;
}

double FXAI_AuditRollingCorr(const double &a[],
                             const double &b[],
                             const int start_idx,
                             const int width)
{
   int n = ArraySize(a);
   if(n != ArraySize(b) || width < 4 || start_idx < 0 || start_idx + width >= n)
      return 0.0;

   double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
   int used = 0;
   for(int k=0; k<width; k++)
   {
      double ra = FXAI_SafeReturn(a, start_idx + k, start_idx + k + 1);
      double rb = FXAI_SafeReturn(b, start_idx + k, start_idx + k + 1);
      sa += ra;
      sb += rb;
      saa += ra * ra;
      sbb += rb * rb;
      sab += ra * rb;
      used++;
   }
   if(used < 4) return 0.0;
   double ma = sa / (double)used;
   double mb = sb / (double)used;
   double va = saa / (double)used - ma * ma;
   double vb = sbb / (double)used - mb * mb;
   double cov = sab / (double)used - ma * mb;
   if(va <= 1e-12 || vb <= 1e-12) return 0.0;
   return FXAI_Clamp(cov / MathSqrt(va * vb), -1.0, 1.0);
}

void FXAI_AuditReverseChronoToSeries(const datetime &src_time[],
                                     const double &src_open[],
                                     const double &src_high[],
                                     const double &src_low[],
                                     const double &src_close[],
                                     const int &src_spread[],
                                     datetime &dst_time[],
                                     double &dst_open[],
                                     double &dst_high[],
                                     double &dst_low[],
                                     double &dst_close[],
                                     int &dst_spread[])
{
   int n = ArraySize(src_close);
   ArrayResize(dst_time, n);
   ArrayResize(dst_open, n);
   ArrayResize(dst_high, n);
   ArrayResize(dst_low, n);
   ArrayResize(dst_close, n);
   ArrayResize(dst_spread, n);
   ArraySetAsSeries(dst_time, true);
   ArraySetAsSeries(dst_open, true);
   ArraySetAsSeries(dst_high, true);
   ArraySetAsSeries(dst_low, true);
   ArraySetAsSeries(dst_close, true);
   ArraySetAsSeries(dst_spread, true);

   for(int i=0; i<n; i++)
   {
      int src = n - 1 - i;
      dst_time[i] = src_time[src];
      dst_open[i] = src_open[src];
      dst_high[i] = src_high[src];
      dst_low[i] = src_low[src];
      dst_close[i] = src_close[src];
      dst_spread[i] = src_spread[src];
   }
}

void FXAI_AuditReverseCloseSeries(const datetime &src_time[],
                                  const double &src_close[],
                                  datetime &dst_time[],
                                  double &dst_close[])
{
   int n = ArraySize(src_close);
   ArrayResize(dst_time, n);
   ArrayResize(dst_close, n);
   ArraySetAsSeries(dst_time, true);
   ArraySetAsSeries(dst_close, true);
   for(int i=0; i<n; i++)
   {
      int src = n - 1 - i;
      dst_time[i] = src_time[src];
      dst_close[i] = src_close[src];
   }
}

void FXAI_AuditAggregateCloseTF(const datetime &src_time_chrono[],
                                const double &src_open_chrono[],
                                const double &src_high_chrono[],
                                const double &src_low_chrono[],
                                const double &src_close_chrono[],
                                const int step,
                                datetime &out_time_series[],
                                double &out_close_series[])
{
   int n = ArraySize(src_close_chrono);
   int bars = (step > 0 ? n / step : 0);
   if(bars <= 0)
   {
      ArrayResize(out_time_series, 0);
      ArrayResize(out_close_series, 0);
      return;
   }

   datetime tmp_time[];
   double tmp_close[];
   ArrayResize(tmp_time, bars);
   ArrayResize(tmp_close, bars);

   for(int b=0; b<bars; b++)
   {
      int start = b * step;
      int end = start + step - 1;
      if(end >= n) end = n - 1;
      tmp_time[b] = src_time_chrono[end];
      tmp_close[b] = src_close_chrono[end];
   }

   FXAI_AuditReverseCloseSeries(tmp_time, tmp_close, out_time_series, out_close_series);
}

void FXAI_AuditBuildContextFeatures(const double &main_close[],
                                    const double &ctx1_close[],
                                    const double &ctx2_close[],
                                    const double &ctx3_close[],
                                    double &ctx_mean_arr[],
                                    double &ctx_std_arr[],
                                    double &ctx_up_arr[],
                                    double &ctx_extra_arr[])
{
   int n = ArraySize(main_close);
   ArrayResize(ctx_mean_arr, n);
   ArrayResize(ctx_std_arr, n);
   ArrayResize(ctx_up_arr, n);
   ArrayResize(ctx_extra_arr, n * FXAI_CONTEXT_EXTRA_FEATS);
   ArraySetAsSeries(ctx_mean_arr, true);
   ArraySetAsSeries(ctx_std_arr, true);
   ArraySetAsSeries(ctx_up_arr, true);
   ArrayInitialize(ctx_extra_arr, 0.0);

   for(int i=0; i<n; i++)
   {
      double main_ret = FXAI_SafeReturn(main_close, i, i + 1);
      double ret[3];
      double lag[3];
      ret[0] = FXAI_SafeReturn(ctx1_close, i, i + 1);
      ret[1] = FXAI_SafeReturn(ctx2_close, i, i + 1);
      ret[2] = FXAI_SafeReturn(ctx3_close, i, i + 1);
      lag[0] = FXAI_SafeReturn(ctx1_close, i + 1, i + 2);
      lag[1] = FXAI_SafeReturn(ctx2_close, i + 1, i + 2);
      lag[2] = FXAI_SafeReturn(ctx3_close, i + 1, i + 2);

      double sum = 0.0;
      double sum2 = 0.0;
      int up = 0;
      for(int s=0; s<3; s++)
      {
         sum += ret[s];
         sum2 += ret[s] * ret[s];
         if(ret[s] > 0.0) up++;
      }
      double mean = sum / 3.0;
      double var = sum2 / 3.0 - mean * mean;
      if(var < 0.0) var = 0.0;
      ctx_mean_arr[i] = mean;
      ctx_std_arr[i] = MathSqrt(var);
      ctx_up_arr[i] = (double)up / 3.0;

      double corr1 = FXAI_AuditRollingCorr(main_close, ctx1_close, i, 16);
      double corr2 = FXAI_AuditRollingCorr(main_close, ctx2_close, i, 16);
      double corr3 = FXAI_AuditRollingCorr(main_close, ctx3_close, i, 16);

      FXAI_SetContextExtraValue(ctx_extra_arr, i, 0, ret[0]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 1, lag[0]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 2, ret[0] - main_ret);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 3, corr1);

      FXAI_SetContextExtraValue(ctx_extra_arr, i, 4, ret[1]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 5, lag[1]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 6, ret[1] - main_ret);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 7, corr2);

      FXAI_SetContextExtraValue(ctx_extra_arr, i, 8, ret[2]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 9, lag[2]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 10, ret[2] - main_ret);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 11, corr3);
   }
}

bool FXAI_AuditGenerateScenarioSeries(const FXAIAuditScenarioSpec &spec,
                                      const int bars,
                                      const ulong seed,
                                      const double point,
                                      datetime &time_series[],
                                      double &open_series[],
                                      double &high_series[],
                                      double &low_series[],
                                      double &close_series[],
                                      int &spread_series[],
                                      datetime &time_m5[],
                                      double &close_m5[],
                                      int &map_m5[],
                                      datetime &time_m15[],
                                      double &close_m15[],
                                      int &map_m15[],
                                      datetime &time_m30[],
                                      double &close_m30[],
                                      int &map_m30[],
                                      datetime &time_h1[],
                                      double &close_h1[],
                                      int &map_h1[],
                                      double &ctx_mean_arr[],
                                      double &ctx_std_arr[],
                                      double &ctx_up_arr[],
                                      double &ctx_extra_arr[])
{
   if(bars < 512 || point <= 0.0) return false;

   CFXAIAuditRng rng;
   rng.Seed(seed ^ ((ulong)(spec.id + 1) * (ulong)2654435761));

   datetime chrono_time[];
   double chrono_open[];
   double chrono_high[];
   double chrono_low[];
   double chrono_close[];
   int chrono_spread[];
   double ctx1[];
   double ctx2[];
   double ctx3[];
   ArrayResize(chrono_time, bars);
   ArrayResize(chrono_open, bars);
   ArrayResize(chrono_high, bars);
   ArrayResize(chrono_low, bars);
   ArrayResize(chrono_close, bars);
   ArrayResize(chrono_spread, bars);
   ArrayResize(ctx1, bars);
   ArrayResize(ctx2, bars);
   ArrayResize(ctx3, bars);

   datetime t0 = D'2024.01.01 00:00';
   double anchor = 1.10000;
   double prev = anchor;
   double prev_sigma = spec.sigma_per_bar;
   double ctx_prev1 = anchor * 0.97;
   double ctx_prev2 = anchor * 1.03;
   double ctx_prev3 = anchor * 0.91;

   for(int k=0; k<bars; k++)
   {
      double sigma = spec.sigma_per_bar;
      if(spec.vol_cluster > 0.0)
      {
         sigma = MathMax(1e-6, spec.vol_cluster * prev_sigma + (1.0 - spec.vol_cluster) * spec.sigma_per_bar * (0.5 + 1.5 * rng.NextUnit()));
         prev_sigma = sigma;
      }

      double drift = spec.drift_per_bar;
      if(spec.id == 7 && k > bars / 2)
      {
         drift = -spec.drift_per_bar * 1.25;
         sigma *= 1.8;
      }

      double ret = drift + sigma * rng.NextNormal();
      if(spec.mean_revert_strength > 0.0)
         ret += spec.mean_revert_strength * ((anchor - prev) / MathMax(prev, point));
      if(spec.spike_prob > 0.0 && rng.NextUnit() < spec.spike_prob)
         ret += spec.spike_scale * sigma * rng.NextNormal();

      if(spec.id == 5)
         ret = MathAbs(drift) + 0.10 * sigma * MathAbs(rng.NextNormal());
      else if(spec.id == 6)
         ret = -(MathAbs(drift) + 0.10 * sigma * MathAbs(rng.NextNormal()));

      double op = prev;
      double cl = prev * (1.0 + ret);
      if(cl <= point) cl = prev + 10.0 * point;
      double wick = MathMax(MathAbs(cl - op) * (0.30 + 0.40 * rng.NextUnit()), point * (2.0 + 8.0 * rng.NextUnit()));
      double hi = MathMax(op, cl) + wick;
      double lo = MathMin(op, cl) - wick;
      if(lo <= point) lo = point;

      chrono_time[k] = t0 + (datetime)(60 * k);
      chrono_open[k] = op;
      chrono_high[k] = hi;
      chrono_low[k] = lo;
      chrono_close[k] = cl;
      chrono_spread[k] = (int)MathMax(1.0, MathRound(spec.spread_points * (0.85 + 0.30 * rng.NextUnit())));

      double ctx_noise1 = 0.35 * sigma * rng.NextNormal();
      double ctx_noise2 = 0.45 * sigma * rng.NextNormal();
      double ctx_noise3 = 0.55 * sigma * rng.NextNormal();
      double ctx_ret1 = 0.80 * ret + ctx_noise1;
      double ctx_ret2 = -0.45 * ret + ctx_noise2;
      double ctx_ret3 = 0.35 * ret + 0.50 * ctx_noise3;
      ctx_prev1 *= (1.0 + ctx_ret1);
      ctx_prev2 *= (1.0 + ctx_ret2);
      ctx_prev3 *= (1.0 + ctx_ret3);
      if(ctx_prev1 <= point) ctx_prev1 = op;
      if(ctx_prev2 <= point) ctx_prev2 = op;
      if(ctx_prev3 <= point) ctx_prev3 = op;
      ctx1[k] = ctx_prev1;
      ctx2[k] = ctx_prev2;
      ctx3[k] = ctx_prev3;

      prev = cl;
   }

   FXAI_AuditReverseChronoToSeries(chrono_time,
                                   chrono_open,
                                   chrono_high,
                                   chrono_low,
                                   chrono_close,
                                   chrono_spread,
                                   time_series,
                                   open_series,
                                   high_series,
                                   low_series,
                                   close_series,
                                   spread_series);

   datetime ctx1_time[];
   datetime ctx2_time[];
   datetime ctx3_time[];
   double ctx1_series[];
   double ctx2_series[];
   double ctx3_series[];
   FXAI_AuditReverseCloseSeries(chrono_time, ctx1, ctx1_time, ctx1_series);
   FXAI_AuditReverseCloseSeries(chrono_time, ctx2, ctx2_time, ctx2_series);
   FXAI_AuditReverseCloseSeries(chrono_time, ctx3, ctx3_time, ctx3_series);

   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 5, time_m5, close_m5);
   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 15, time_m15, close_m15);
   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 30, time_m30, close_m30);
   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 60, time_h1, close_h1);

   FXAI_BuildAlignedIndexMap(time_series, time_m5, 2 * PeriodSeconds(PERIOD_M5), map_m5);
   FXAI_BuildAlignedIndexMap(time_series, time_m15, 2 * PeriodSeconds(PERIOD_M15), map_m15);
   FXAI_BuildAlignedIndexMap(time_series, time_m30, 2 * PeriodSeconds(PERIOD_M30), map_m30);
   FXAI_BuildAlignedIndexMap(time_series, time_h1, 2 * PeriodSeconds(PERIOD_H1), map_h1);

   FXAI_AuditBuildContextFeatures(close_series,
                                  ctx1_series,
                                  ctx2_series,
                                  ctx3_series,
                                  ctx_mean_arr,
                                  ctx_std_arr,
                                  ctx_up_arr,
                                  ctx_extra_arr);
   return true;
}

bool FXAI_AuditBuildSample(const int i,
                           const int horizon_minutes,
                           const double point,
                           const double ev_threshold_points,
                           const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                           const datetime &time_arr[],
                           const double &open_arr[],
                           const double &high_arr[],
                           const double &low_arr[],
                           const double &close_arr[],
                           const int &spread_arr[],
                           const datetime &time_m5[],
                           const double &close_m5[],
                           const int &map_m5[],
                           const datetime &time_m15[],
                           const double &close_m15[],
                           const int &map_m15[],
                           const datetime &time_m30[],
                           const double &close_m30[],
                           const int &map_m30[],
                           const datetime &time_h1[],
                           const double &close_h1[],
                           const int &map_h1[],
                           const double &ctx_mean_arr[],
                           const double &ctx_std_arr[],
                           const double &ctx_up_arr[],
                           const double &ctx_extra_arr[],
                           FXAIAIContextV4 &ctx,
                           int &label_class,
                           double &move_points,
                           double &sample_weight,
                           double &x[])
{
   int n = ArraySize(close_arr);
   if(i < 0 || i >= n) return false;
   if(i - horizon_minutes < 0) return false;

   FXAIDataSnapshot snapshot;
   snapshot.symbol = "SYNTH";
   snapshot.bar_time = time_arr[i];
   snapshot.point = point;
   snapshot.spread_points = FXAI_GetSpreadAtIndex(i, spread_arr, 1.0);
   snapshot.commission_points = 0.0;
   snapshot.min_move_points = snapshot.spread_points;

   double feat[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(i,
                                 snapshot.spread_points,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
                                 time_m5,
                                 close_m5,
                                 map_m5,
                                 time_m15,
                                 close_m15,
                                 map_m15,
                                 time_m30,
                                 close_m30,
                                 map_m30,
                                 time_h1,
                                 close_h1,
                                 map_h1,
                                 FXAI_AuditGetArrayValue(ctx_mean_arr, i, 0.0),
                                 FXAI_AuditGetArrayValue(ctx_std_arr, i, 0.0),
                                 FXAI_AuditGetArrayValue(ctx_up_arr, i, 0.5),
                                 ctx_extra_arr,
                                 norm_method,
                                 feat))
      return false;

   bool need_prev = FXAI_FeatureNormNeedsPrevious(norm_method);
   bool has_prev = false;
   double feat_prev[FXAI_AI_FEATURES];
   for(int f=0; f<FXAI_AI_FEATURES; f++) feat_prev[f] = 0.0;
   if(need_prev && (i + 1) < n)
   {
      has_prev = FXAI_ComputeFeatureVector(i + 1,
                                           FXAI_GetSpreadAtIndex(i + 1, spread_arr, snapshot.spread_points),
                                           time_arr,
                                           open_arr,
                                           high_arr,
                                           low_arr,
                                           close_arr,
                                           time_m5,
                                           close_m5,
                                           map_m5,
                                           time_m15,
                                           close_m15,
                                           map_m15,
                                           time_m30,
                                           close_m30,
                                           map_m30,
                                           time_h1,
                                           close_h1,
                                           map_h1,
                                           FXAI_AuditGetArrayValue(ctx_mean_arr, i + 1, 0.0),
                                           FXAI_AuditGetArrayValue(ctx_std_arr, i + 1, 0.0),
                                           FXAI_AuditGetArrayValue(ctx_up_arr, i + 1, 0.5),
                                           ctx_extra_arr,
                                           norm_method,
                                           feat_prev);
   }

   double feat_norm[FXAI_AI_FEATURES];
   FXAI_ApplyFeatureNormalization(norm_method, feat, feat_prev, has_prev, snapshot.bar_time, feat_norm);
   ArrayResize(x, FXAI_AI_WEIGHTS);
   FXAI_BuildInputVector(feat_norm, x);

   move_points = FXAI_MovePoints(close_arr[i], close_arr[i - horizon_minutes], point);
   double cost_points = snapshot.spread_points;
   label_class = FXAI_BuildEVClassLabel(move_points, cost_points, ev_threshold_points);
   sample_weight = FXAI_Clamp(FXAI_MoveEdgeWeight(move_points, cost_points), 0.25, 4.0);

   double spread_ref = FXAI_AuditGetIntArrayMean(spread_arr, i, 64, snapshot.spread_points);
   double vol_ref = FXAI_RollingAbsReturn(close_arr, i, 64);
   double vol_proxy = MathAbs(feat[5]);
   ctx.api_version = FXAI_API_VERSION_V4;
   ctx.regime_id = FXAI_AuditGetStaticRegimeId(snapshot.bar_time, snapshot.spread_points, spread_ref, vol_proxy, vol_ref);
   ctx.session_bucket = FXAI_DeriveSessionBucket(snapshot.bar_time);
   ctx.horizon_minutes = horizon_minutes;
   ctx.feature_schema_id = 1;
   ctx.normalization_method_id = (int)norm_method;
   ctx.sequence_bars = 1;
   ctx.cost_points = cost_points;
   ctx.min_move_points = cost_points;
   ctx.point_value = point;
   ctx.sample_time = snapshot.bar_time;
   return true;
}

void FXAI_AuditComparePredictions(const FXAIAIPredictionV4 &a,
                                  const FXAIAIPredictionV4 &b,
                                  double &delta_out)
{
   delta_out = 0.0;
   for(int c=0; c<3; c++)
      delta_out += MathAbs(a.class_probs[c] - b.class_probs[c]);
   delta_out += 0.10 * MathAbs(a.move_mean_points - b.move_mean_points);
   delta_out += 0.05 * MathAbs(a.confidence - b.confidence);
   delta_out += 0.05 * MathAbs(a.reliability - b.reliability);
}

void FXAI_AuditFinalizeMetrics(FXAIAuditScenarioMetrics &m)
{
   if(m.samples_total > 0)
   {
      m.skip_ratio = (double)m.skip_count / (double)m.samples_total;
      m.active_ratio = (double)(m.buy_count + m.sell_count) / (double)m.samples_total;
   }
   int active = m.buy_count + m.sell_count;
   if(active > 0)
      m.bias_abs = MathAbs((double)m.buy_count - (double)m.sell_count) / (double)active;
   if(m.directional_eval_count > 0)
   {
      double avg_conf = m.dir_conf_sum / (double)m.directional_eval_count;
      double avg_hit = m.dir_hit_sum / (double)m.directional_eval_count;
      m.conf_drift = MathAbs(avg_conf - avg_hit);
   }

   double score = 100.0;
   if(m.invalid_preds > 0) score -= 35.0;
   if(m.skip_ratio < 0.45 && m.scenario == "random_walk") score -= 18.0;
   if(m.active_ratio > 0.80 && m.scenario == "random_walk") score -= 12.0;
   if(m.trend_alignment_count > 0)
   {
      double align = m.trend_alignment_sum / (double)m.trend_alignment_count;
      if((m.scenario == "drift_up" || m.scenario == "drift_down" || m.scenario == "monotonic_up" || m.scenario == "monotonic_down") && align < 0.20)
         score -= 18.0;
   }
   if(m.conf_drift > 0.22) score -= 10.0;
   if(m.reset_delta > 0.30) score -= 12.0;
   if(m.sequence_delta < 0.005 && m.sequence_delta >= 0.0) score -= 6.0;
   if(m.move_sum <= 0.0) score -= 8.0;
   if(score < 0.0) score = 0.0;
   m.score = score;

   if(m.invalid_preds > 0) m.issue_flags |= FXAI_AUDIT_ISSUE_INVALID_PRED;
   if(m.scenario == "random_walk" && (m.skip_ratio < 0.55 || m.active_ratio > 0.70))
      m.issue_flags |= FXAI_AUDIT_ISSUE_OVERTRADES_NOISE;
   if((m.scenario == "drift_up" || m.scenario == "drift_down" || m.scenario == "monotonic_up" || m.scenario == "monotonic_down") &&
      m.trend_alignment_count > 0 && (m.trend_alignment_sum / (double)m.trend_alignment_count) < 0.25)
      m.issue_flags |= FXAI_AUDIT_ISSUE_MISSES_TREND;
   if(m.conf_drift > 0.22) m.issue_flags |= FXAI_AUDIT_ISSUE_CALIBRATION_DRIFT;
   if(m.reset_delta > 0.30) m.issue_flags |= FXAI_AUDIT_ISSUE_RESET_DRIFT;
   if(m.sequence_delta >= 0.0 && m.sequence_delta < 0.005) m.issue_flags |= FXAI_AUDIT_ISSUE_SEQUENCE_WEAK;
   if(m.move_sum <= 0.0) m.issue_flags |= FXAI_AUDIT_ISSUE_DEAD_OUTPUT;
   if(m.scenario == "random_walk" && m.bias_abs > 0.85 && active > 24)
      m.issue_flags |= FXAI_AUDIT_ISSUE_SIDE_COLLAPSE;
}

bool FXAI_AuditRunScenario(CFXAIAIRegistry &registry,
                           const int ai_idx,
                           const FXAIAuditScenarioSpec &spec,
                           const int bars,
                           const int horizon_minutes,
                           const ulong seed,
                           const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                           FXAIAuditScenarioMetrics &out)
{
   CFXAIAIPlugin *plugin = registry.CreateInstance(ai_idx);
   if(plugin == NULL) return false;

   FXAIAIManifestV4 manifest;
   plugin.Describe(manifest);
   FXAI_AuditResetMetrics(out, ai_idx, manifest.ai_name, manifest.family, spec.name, bars);

   datetime time_arr[];
   double open_arr[];
   double high_arr[];
   double low_arr[];
   double close_arr[];
   int spread_arr[];
   datetime time_m5[];
   double close_m5[];
   int map_m5[];
   datetime time_m15[];
   double close_m15[];
   int map_m15[];
   datetime time_m30[];
   double close_m30[];
   int map_m30[];
   datetime time_h1[];
   double close_h1[];
   int map_h1[];
   double ctx_mean_arr[];
   double ctx_std_arr[];
   double ctx_up_arr[];
   double ctx_extra_arr[];
   double point = 0.0001;

   if(!FXAI_AuditGenerateScenarioSeries(spec,
                                        bars,
                                        seed + ((ulong)(ai_idx + 1) * (ulong)1315423911),
                                        point,
                                        time_arr,
                                        open_arr,
                                        high_arr,
                                        low_arr,
                                        close_arr,
                                        spread_arr,
                                        time_m5,
                                        close_m5,
                                        map_m5,
                                        time_m15,
                                        close_m15,
                                        map_m15,
                                        time_m30,
                                        close_m30,
                                        map_m30,
                                        time_h1,
                                        close_h1,
                                        map_h1,
                                        ctx_mean_arr,
                                        ctx_std_arr,
                                        ctx_up_arr,
                                        ctx_extra_arr))
   {
      delete plugin;
      return false;
   }

   FXAI_ResetNormalizationWindows(192);
   if(plugin.SupportsSyntheticSeries())
      plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);

   FXAIAIHyperParams hp;
   FXAI_AuditDefaultHyperParams(ai_idx, hp);
   plugin.EnsureInitialized(hp);
   int seq_bars = FXAI_ResolveManifestSequenceBars(manifest, horizon_minutes);

   int n = ArraySize(close_arr);
   int start_idx = horizon_minutes + 1;
   int end_idx = n - 220;
   if(end_idx <= start_idx) end_idx = n - 32;
   if(end_idx <= start_idx) end_idx = n - 2;

   FXAIAIPredictionV4 held_pred_reset;
   FXAIAIPredictRequestV4 held_req;
   held_req.valid = false;
   bool held_req_ready = false;

   for(int i=start_idx; i<end_idx; i++)
   {
      FXAIAIContextV4 ctx;
      int label_class = (int)FXAI_LABEL_SKIP;
      double move_points = 0.0;
      double sample_weight = 1.0;
      double x[];
      if(!FXAI_AuditBuildSample(i,
                                horizon_minutes,
                                point,
                                0.25,
                                norm_method,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                spread_arr,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_m30,
                                close_m30,
                                map_m30,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_arr,
                                ctx_std_arr,
                                ctx_up_arr,
                                ctx_extra_arr,
                                ctx,
                                label_class,
                                move_points,
                                sample_weight,
                                x))
         continue;

      ctx.sequence_bars = seq_bars;
      out.samples_total++;
      if(label_class == (int)FXAI_LABEL_BUY) out.true_buy_count++;
      else if(label_class == (int)FXAI_LABEL_SELL) out.true_sell_count++;
      else out.true_skip_count++;

      FXAIAIPredictRequestV4 req;
      req.valid = true;
      req.ctx = ctx;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++) req.x[k] = x[k];

      FXAIAIPredictionV4 pred;
      bool ok = FXAI_PredictViaV4(*plugin, req, hp, pred);
      string pred_reason = "";
      bool pred_valid = FXAI_ValidatePredictionV4(pred, pred_reason);
      if(!ok || !pred_valid)
      {
         out.invalid_preds++;
      }
      else
      {
         out.valid_preds++;
         int decision = FXAI_AuditDecisionFromPred(pred);
         if(decision == (int)FXAI_LABEL_BUY) out.buy_count++;
         else if(decision == (int)FXAI_LABEL_SELL) out.sell_count++;
         else out.skip_count++;

         if(decision == label_class) out.exact_match_count++;

         if(spec.name == "drift_up" || spec.name == "monotonic_up")
         {
            if(decision == (int)FXAI_LABEL_BUY) out.trend_alignment_sum += 1.0;
            else if(decision == (int)FXAI_LABEL_SELL) out.trend_alignment_sum -= 1.0;
            out.trend_alignment_count++;
         }
         else if(spec.name == "drift_down" || spec.name == "monotonic_down")
         {
            if(decision == (int)FXAI_LABEL_SELL) out.trend_alignment_sum += 1.0;
            else if(decision == (int)FXAI_LABEL_BUY) out.trend_alignment_sum -= 1.0;
            out.trend_alignment_count++;
         }

         out.conf_sum += pred.confidence;
         out.rel_sum += pred.reliability;
         out.move_sum += pred.move_mean_points;

         if(decision != (int)FXAI_LABEL_SKIP)
         {
            out.directional_eval_count++;
            double dir_conf = MathMax(pred.class_probs[(int)FXAI_LABEL_BUY], pred.class_probs[(int)FXAI_LABEL_SELL]);
            out.dir_conf_sum += dir_conf;
            bool dir_ok = ((decision == (int)FXAI_LABEL_BUY && label_class == (int)FXAI_LABEL_BUY) ||
                           (decision == (int)FXAI_LABEL_SELL && label_class == (int)FXAI_LABEL_SELL));
            if(dir_ok) out.directional_correct_count++;
            out.dir_hit_sum += (dir_ok ? 1.0 : 0.0);
         }
      }

      FXAIAITrainRequestV4 train_req;
      train_req.valid = true;
      train_req.ctx = ctx;
      train_req.label_class = label_class;
      train_req.move_points = move_points;
      train_req.sample_weight = sample_weight;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++) train_req.x[k] = x[k];
      FXAI_TrainViaV4(*plugin, train_req, hp);

      if(!held_req_ready && i > start_idx + 128)
      {
         held_req = req;
         held_req_ready = true;
      }
   }

   if(held_req_ready)
   {
      FXAI_PredictViaV4(*plugin, held_req, hp, held_pred_reset);
      plugin.ResetState((int)FXAI_RESET_MANUAL, held_req.ctx.sample_time);
      if(plugin.SupportsSyntheticSeries())
         plugin.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);
      FXAIAIPredictionV4 pred_after_reset;
      FXAI_PredictViaV4(*plugin, held_req, hp, pred_after_reset);

      CFXAIAIPlugin *fresh = registry.CreateInstance(ai_idx);
      if(fresh != NULL)
      {
         fresh.EnsureInitialized(hp);
         if(fresh.SupportsSyntheticSeries())
            fresh.SetSyntheticSeries(time_arr, open_arr, high_arr, low_arr, close_arr);
         FXAIAIPredictionV4 pred_fresh;
         FXAI_PredictViaV4(*fresh, held_req, hp, pred_fresh);
         FXAI_AuditComparePredictions(pred_after_reset, pred_fresh, out.reset_delta);
         delete fresh;
      }

      if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT) ||
         FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL))
      {
         FXAIAIPredictRequestV4 seq_short = held_req;
         seq_short.ctx.sequence_bars = 1;
         FXAIAIPredictRequestV4 seq_long = held_req;
         seq_long.ctx.sequence_bars = seq_bars;
         FXAIAIPredictionV4 pred_short;
         FXAIAIPredictionV4 pred_long;
         FXAI_PredictViaV4(*plugin, seq_short, hp, pred_short);
         FXAI_PredictViaV4(*plugin, seq_long, hp, pred_long);
         FXAI_AuditComparePredictions(pred_short, pred_long, out.sequence_delta);
      }
      else
      {
         out.sequence_delta = -1.0;
      }
   }
   else
   {
      out.reset_delta = -1.0;
      out.sequence_delta = -1.0;
   }

   FXAI_AuditFinalizeMetrics(out);
   plugin.ClearSyntheticSeries();
   delete plugin;
   return true;
}

bool FXAI_AuditWriteHeader(const int handle)
{
   return FileWrite(handle,
                    "ai_id\tai_name\tfamily\tscenario\tbars_total\tsamples_total\tvalid_preds\tinvalid_preds\tbuy_count\tsell_count\tskip_count\ttrue_buy_count\ttrue_sell_count\ttrue_skip_count\texact_match_count\tdirectional_eval_count\tdirectional_correct_count\tskip_ratio\tactive_ratio\tbias_abs\tconf_drift\treset_delta\tsequence_delta\tscore\tissue_flags\tavg_conf\tavg_rel\tavg_move\ttrend_align") > 0;
}

bool FXAI_AuditWriteMetrics(const int handle,
                            const FXAIAuditScenarioMetrics &m)
{
   double avg_conf = (m.valid_preds > 0 ? m.conf_sum / (double)m.valid_preds : 0.0);
   double avg_rel = (m.valid_preds > 0 ? m.rel_sum / (double)m.valid_preds : 0.0);
   double avg_move = (m.valid_preds > 0 ? m.move_sum / (double)m.valid_preds : 0.0);
   double trend_align = (m.trend_alignment_count > 0 ? m.trend_alignment_sum / (double)m.trend_alignment_count : 0.0);
   return FileWrite(handle,
                    IntegerToString(m.ai_id) + "\t" +
                    m.ai_name + "\t" +
                    IntegerToString(m.family) + "\t" +
                    m.scenario + "\t" +
                    IntegerToString(m.bars_total) + "\t" +
                    IntegerToString(m.samples_total) + "\t" +
                    IntegerToString(m.valid_preds) + "\t" +
                    IntegerToString(m.invalid_preds) + "\t" +
                    IntegerToString(m.buy_count) + "\t" +
                    IntegerToString(m.sell_count) + "\t" +
                    IntegerToString(m.skip_count) + "\t" +
                    IntegerToString(m.true_buy_count) + "\t" +
                    IntegerToString(m.true_sell_count) + "\t" +
                    IntegerToString(m.true_skip_count) + "\t" +
                    IntegerToString(m.exact_match_count) + "\t" +
                    IntegerToString(m.directional_eval_count) + "\t" +
                    IntegerToString(m.directional_correct_count) + "\t" +
                    DoubleToString(m.skip_ratio, 6) + "\t" +
                    DoubleToString(m.active_ratio, 6) + "\t" +
                    DoubleToString(m.bias_abs, 6) + "\t" +
                    DoubleToString(m.conf_drift, 6) + "\t" +
                    DoubleToString(m.reset_delta, 6) + "\t" +
                    DoubleToString(m.sequence_delta, 6) + "\t" +
                    DoubleToString(m.score, 4) + "\t" +
                    IntegerToString(m.issue_flags) + "\t" +
                    DoubleToString(avg_conf, 6) + "\t" +
                    DoubleToString(avg_rel, 6) + "\t" +
                    DoubleToString(avg_move, 6) + "\t" +
                    DoubleToString(trend_align, 6)) > 0;
}

#endif // __FXAI_AUDIT_CORE_MQH__
