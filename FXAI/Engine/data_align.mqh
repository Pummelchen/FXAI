#ifndef __FXAI_DATA_ALIGN_MQH__
#define __FXAI_DATA_ALIGN_MQH__

double FXAI_MovePoints(const double price_now,
                      const double price_future,
                      const double point)
{
   if(point <= 0.0) return 0.0;
   return (price_future - price_now) / point;
}

int FXAI_BuildEVClassLabel(const double move_points,
                          const double roundtrip_cost_points,
                          const double ev_threshold_points)
{
   double ev_min = (ev_threshold_points < 0.0 ? 0.0 : ev_threshold_points);

   double buy_ev = move_points - roundtrip_cost_points;
   double sell_ev = -move_points - roundtrip_cost_points;

   if(buy_ev >= ev_min && buy_ev > sell_ev) return (int)FXAI_LABEL_BUY;
   if(sell_ev >= ev_min && sell_ev > buy_ev) return (int)FXAI_LABEL_SELL;
   return (int)FXAI_LABEL_SKIP;
}

int FXAI_FindAlignedIndex(const datetime &time_arr[],
                         const datetime ref_time,
                         const int max_lag_seconds)
{
   int n = ArraySize(time_arr);
   if(n <= 0) return -1;
   if(ref_time <= 0) return -1;

   int lo = 0;
   int hi = n - 1;
   int ans = -1;

   // descending timeseries: first index where bar_time <= ref_time
   while(lo <= hi)
   {
      int mid = (lo + hi) / 2;
      datetime t = time_arr[mid];
      if(t <= ref_time)
      {
         ans = mid;
         hi = mid - 1;
      }
      else
      {
         lo = mid + 1;
      }
   }

   if(ans < 0) return -1;
   if(max_lag_seconds <= 0) return ans;

   long lag = (long)(ref_time - time_arr[ans]);
   if(lag < 0 || lag > (long)max_lag_seconds)
      return -1;

   return ans;
}

double FXAI_AlignedFreshnessWeight(const datetime &time_arr[],
                                  const int idx,
                                  const datetime ref_time,
                                  const int max_lag_seconds)
{
   int n = ArraySize(time_arr);
   if(n <= 0) return 0.0;
   if(idx < 0 || idx >= n) return 0.0;
   if(ref_time <= 0) return 0.0;
   if(max_lag_seconds <= 0) return 1.0;

   long lag = (long)(ref_time - time_arr[idx]);
   if(lag < 0 || lag > (long)max_lag_seconds) return 0.0;

   double w = 1.0 - ((double)lag / (double)max_lag_seconds);
   return FXAI_Clamp(w, 0.0, 1.0);
}

void FXAI_BuildAlignedIndexMap(const datetime &ref_time_arr[],
                              const datetime &target_time_arr[],
                              const int max_lag_seconds,
                              int &out_idx_arr[])
{
   int n_ref = ArraySize(ref_time_arr);
   FXAI_BuildAlignedIndexMapRange(ref_time_arr, target_time_arr, max_lag_seconds, n_ref - 1, out_idx_arr);
}

void FXAI_BuildAlignedIndexMapRange(const datetime &ref_time_arr[],
                                   const datetime &target_time_arr[],
                                   const int max_lag_seconds,
                                   const int upto_index,
                                   int &out_idx_arr[])
{
   int n_ref = ArraySize(ref_time_arr);
   int cur_sz = ArraySize(out_idx_arr);
   if(cur_sz != n_ref) ArrayResize(out_idx_arr, n_ref);
   ArraySetAsSeries(out_idx_arr, true);

   if(n_ref <= 0)
      return;

   int upto = upto_index;
   if(upto < 0) upto = 0;
   if(upto >= n_ref) upto = n_ref - 1;

   // Always clear full map to prevent stale indices when caller changes
   // range size across calls.
   for(int i=0; i<n_ref; i++)
      out_idx_arr[i] = -1;

   int n_tgt = ArraySize(target_time_arr);
   if(n_tgt <= 0) return;

   int j = 0;
   for(int i=0; i<=upto; i++)
   {
      datetime t_ref = ref_time_arr[i];
      if(t_ref <= 0) continue;

      while(j < n_tgt && target_time_arr[j] > t_ref)
         j++;

      if(j >= n_tgt)
         break;

      datetime t_tgt = target_time_arr[j];
      long lag = (long)(t_ref - t_tgt);
      if(lag < 0) continue;
      if(max_lag_seconds > 0 && lag > (long)max_lag_seconds) continue;

      out_idx_arr[i] = j;
   }
}


#endif // __FXAI_DATA_ALIGN_MQH__
