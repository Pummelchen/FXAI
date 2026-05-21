#ifndef __FXAI_MARKET_DATA_GATEWAY_MQH__
#define __FXAI_MARKET_DATA_GATEWAY_MQH__

enum ENUM_FXAI_MARKET_DATA_OP
{
   FXAI_MARKET_DATA_OP_NONE = 0,
   FXAI_MARKET_DATA_OP_COPY_RATES_BY_POS = 1,
   FXAI_MARKET_DATA_OP_COPY_RATES_BY_TIME = 2,
   FXAI_MARKET_DATA_OP_COPY_CLOSE_BY_POS = 3,
   FXAI_MARKET_DATA_OP_COPY_TICKS_LATEST = 4,
   FXAI_MARKET_DATA_OP_COPY_TICKS_RANGE = 5,
   FXAI_MARKET_DATA_OP_LIVE_TICK = 6,
   FXAI_MARKET_DATA_OP_BAR_SHIFT = 7,
   FXAI_MARKET_DATA_OP_BAR_TIME = 8,
   FXAI_MARKET_DATA_OP_BAR_OPEN = 9,
   FXAI_MARKET_DATA_OP_BAR_HIGH = 10,
   FXAI_MARKET_DATA_OP_BAR_LOW = 11,
   FXAI_MARKET_DATA_OP_BAR_CLOSE = 12
};

void FXAI_MarketDataResetTick(MqlTick &tick_out)
{
   tick_out.time = 0;
   tick_out.bid = 0.0;
   tick_out.ask = 0.0;
   tick_out.last = 0.0;
   tick_out.volume = 0;
   tick_out.time_msc = 0;
   tick_out.flags = 0;
   tick_out.volume_real = 0.0;
}

void FXAI_MarketDataResetOutputs(int &int_out,
                                 datetime &time_out,
                                 double &double_out,
                                 MqlTick &tick_out,
                                 MqlRates &rates_out[],
                                 double &values_out[],
                                 MqlTick &ticks_out[])
{
   int_out = 0;
   time_out = 0;
   double_out = 0.0;
   FXAI_MarketDataResetTick(tick_out);
   ArrayResize(rates_out, 0);
   ArrayResize(values_out, 0);
   ArrayResize(ticks_out, 0);
}

bool FXAI_MarketDataPull(const ENUM_FXAI_MARKET_DATA_OP op,
                         const string symbol,
                         const ENUM_TIMEFRAMES tf,
                         const int start_pos,
                         const int count,
                         const datetime from_time,
                         const datetime to_time,
                         const ulong from_msc,
                         const ulong to_msc,
                         const int tick_flags,
                         const bool exact,
                         int &int_out,
                         datetime &time_out,
                         double &double_out,
                         MqlTick &tick_out,
                         MqlRates &rates_out[],
                         double &values_out[],
                         MqlTick &ticks_out[])
{
   FXAI_MarketDataResetOutputs(int_out,
                               time_out,
                               double_out,
                               tick_out,
                               rates_out,
                               values_out,
                               ticks_out);

   if(StringLen(symbol) <= 0)
      return false;

   if(!SymbolSelect(symbol, true))
      return false;

   switch(op)
   {
      case FXAI_MARKET_DATA_OP_COPY_RATES_BY_POS:
      {
         if(count <= 0)
            return false;
         ArrayResize(rates_out, count);
         ArraySetAsSeries(rates_out, true);
         int copied = CopyRates(symbol, tf, start_pos, count, rates_out);
         if(copied <= 0)
         {
            ArrayResize(rates_out, 0);
            return false;
         }
         if(copied < count)
            ArrayResize(rates_out, copied);
         int_out = copied;
         return true;
      }

      case FXAI_MARKET_DATA_OP_COPY_RATES_BY_TIME:
      {
         if(from_time <= 0 || to_time <= from_time)
            return false;
         ArraySetAsSeries(rates_out, true);
         int copied = CopyRates(symbol, tf, from_time, to_time, rates_out);
         if(copied <= 0)
         {
            ArrayResize(rates_out, 0);
            return false;
         }
         int_out = copied;
         return true;
      }

      case FXAI_MARKET_DATA_OP_COPY_CLOSE_BY_POS:
      {
         if(count <= 0)
            return false;
         MqlRates rates_tmp[];
         ArrayResize(rates_tmp, count);
         ArraySetAsSeries(rates_tmp, true);
         int copied = CopyRates(symbol, tf, start_pos, count, rates_tmp);
         if(copied <= 0)
            return false;
         ArrayResize(values_out, copied);
         ArraySetAsSeries(values_out, true);
         for(int i=0; i<copied; i++)
            values_out[i] = rates_tmp[i].close;
         int_out = copied;
         return true;
      }

      case FXAI_MARKET_DATA_OP_COPY_TICKS_LATEST:
      {
         if(count <= 0)
            return false;
         int copied = CopyTicks(symbol, ticks_out, tick_flags, from_msc, count);
         if(copied <= 0)
         {
            ArrayResize(ticks_out, 0);
            return false;
         }
         int_out = copied;
         return true;
      }

      case FXAI_MARKET_DATA_OP_COPY_TICKS_RANGE:
      {
         if(to_msc <= from_msc)
            return false;
         int copied = CopyTicksRange(symbol, ticks_out, tick_flags, from_msc, to_msc);
         if(copied <= 0)
         {
            ArrayResize(ticks_out, 0);
            return false;
         }
         int_out = copied;
         return true;
      }

      case FXAI_MARKET_DATA_OP_LIVE_TICK:
      {
         if(!SymbolInfoTick(symbol, tick_out))
         {
            FXAI_MarketDataResetTick(tick_out);
            return false;
         }
         return true;
      }

      case FXAI_MARKET_DATA_OP_BAR_SHIFT:
      {
         if(from_time <= 0)
            return false;
         int shift = iBarShift(symbol, tf, from_time, exact);
         int_out = shift;
         return (shift >= 0);
      }

      case FXAI_MARKET_DATA_OP_BAR_TIME:
      case FXAI_MARKET_DATA_OP_BAR_OPEN:
      case FXAI_MARKET_DATA_OP_BAR_HIGH:
      case FXAI_MARKET_DATA_OP_BAR_LOW:
      case FXAI_MARKET_DATA_OP_BAR_CLOSE:
      {
         MqlRates rate_buf[];
         ArrayResize(rate_buf, 1);
         ArraySetAsSeries(rate_buf, true);
         int copied = CopyRates(symbol, tf, start_pos, 1, rate_buf);
         if(copied <= 0)
            return false;
         int_out = copied;
         time_out = rate_buf[0].time;
         if(op == FXAI_MARKET_DATA_OP_BAR_OPEN) double_out = rate_buf[0].open;
         else if(op == FXAI_MARKET_DATA_OP_BAR_HIGH) double_out = rate_buf[0].high;
         else if(op == FXAI_MARKET_DATA_OP_BAR_LOW) double_out = rate_buf[0].low;
         else if(op == FXAI_MARKET_DATA_OP_BAR_CLOSE) double_out = rate_buf[0].close;
         return true;
      }
   }

   return false;
}

bool FXAI_MarketDataCopyRatesByPos(const string symbol,
                                   const ENUM_TIMEFRAMES tf,
                                   const int start_pos,
                                   const int count,
                                   MqlRates &rates_out[])
{
   int int_out = 0;
   datetime time_out = 0;
   double double_out = 0.0;
   MqlTick tick_out;
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_COPY_RATES_BY_POS,
                              symbol,
                              tf,
                              start_pos,
                              count,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              double_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataCopyRatesByTime(const string symbol,
                                    const ENUM_TIMEFRAMES tf,
                                    const datetime from_time,
                                    const datetime to_time,
                                    MqlRates &rates_out[])
{
   int int_out = 0;
   datetime time_out = 0;
   double double_out = 0.0;
   MqlTick tick_out;
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_COPY_RATES_BY_TIME,
                              symbol,
                              tf,
                              0,
                              0,
                              from_time,
                              to_time,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              double_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataCopyCloseByPos(const string symbol,
                                   const ENUM_TIMEFRAMES tf,
                                   const int start_pos,
                                   const int count,
                                   double &close_out[])
{
   int int_out = 0;
   datetime time_out = 0;
   double double_out = 0.0;
   MqlTick tick_out;
   MqlRates rates_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_COPY_CLOSE_BY_POS,
                              symbol,
                              tf,
                              start_pos,
                              count,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              double_out,
                              tick_out,
                              rates_out,
                              close_out,
                              ticks_out);
}

bool FXAI_MarketDataCopyTicksLatest(const string symbol,
                                    const int tick_flags,
                                    const ulong from_msc,
                                    const int count,
                                    MqlTick &ticks_out[])
{
   int int_out = 0;
   datetime time_out = 0;
   double double_out = 0.0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_COPY_TICKS_LATEST,
                              symbol,
                              PERIOD_CURRENT,
                              0,
                              count,
                              0,
                              0,
                              from_msc,
                              0,
                              tick_flags,
                              false,
                              int_out,
                              time_out,
                              double_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataCopyTicksRange(const string symbol,
                                   const int tick_flags,
                                   const ulong from_msc,
                                   const ulong to_msc,
                                   MqlTick &ticks_out[])
{
   int int_out = 0;
   datetime time_out = 0;
   double double_out = 0.0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_COPY_TICKS_RANGE,
                              symbol,
                              PERIOD_CURRENT,
                              0,
                              0,
                              0,
                              0,
                              from_msc,
                              to_msc,
                              tick_flags,
                              false,
                              int_out,
                              time_out,
                              double_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataGetLatestTick(const string symbol,
                                  MqlTick &tick_out)
{
   int int_out = 0;
   datetime time_out = 0;
   double double_out = 0.0;
   MqlRates rates_out[];
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_LIVE_TICK,
                              symbol,
                              PERIOD_CURRENT,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              double_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataBarShift(const string symbol,
                             const ENUM_TIMEFRAMES tf,
                             const datetime at_time,
                             const bool exact,
                             int &shift_out)
{
   datetime time_out = 0;
   double double_out = 0.0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_BAR_SHIFT,
                              symbol,
                              tf,
                              0,
                              0,
                              at_time,
                              0,
                              0,
                              0,
                              0,
                              exact,
                              shift_out,
                              time_out,
                              double_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataBarTime(const string symbol,
                            const ENUM_TIMEFRAMES tf,
                            const int shift,
                            datetime &bar_time)
{
   int int_out = 0;
   double double_out = 0.0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_BAR_TIME,
                              symbol,
                              tf,
                              shift,
                              1,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              bar_time,
                              double_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataBarOpen(const string symbol,
                            const ENUM_TIMEFRAMES tf,
                            const int shift,
                            double &value_out)
{
   int int_out = 0;
   datetime time_out = 0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_BAR_OPEN,
                              symbol,
                              tf,
                              shift,
                              1,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              value_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataBarHigh(const string symbol,
                            const ENUM_TIMEFRAMES tf,
                            const int shift,
                            double &value_out)
{
   int int_out = 0;
   datetime time_out = 0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_BAR_HIGH,
                              symbol,
                              tf,
                              shift,
                              1,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              value_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataBarLow(const string symbol,
                           const ENUM_TIMEFRAMES tf,
                           const int shift,
                           double &value_out)
{
   int int_out = 0;
   datetime time_out = 0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_BAR_LOW,
                              symbol,
                              tf,
                              shift,
                              1,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              value_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

bool FXAI_MarketDataBarClose(const string symbol,
                             const ENUM_TIMEFRAMES tf,
                             const int shift,
                             double &value_out)
{
   int int_out = 0;
   datetime time_out = 0;
   MqlTick tick_out;
   MqlRates rates_out[];
   double values_out[];
   MqlTick ticks_out[];
   return FXAI_MarketDataPull(FXAI_MARKET_DATA_OP_BAR_CLOSE,
                              symbol,
                              tf,
                              shift,
                              1,
                              0,
                              0,
                              0,
                              0,
                              0,
                              false,
                              int_out,
                              time_out,
                              value_out,
                              tick_out,
                              rates_out,
                              values_out,
                              ticks_out);
}

#endif // __FXAI_MARKET_DATA_GATEWAY_MQH__
