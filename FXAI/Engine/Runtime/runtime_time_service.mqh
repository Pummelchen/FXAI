#ifndef __FXAI_RUNTIME_TIME_SERVICE_MQH__
#define __FXAI_RUNTIME_TIME_SERVICE_MQH__

struct FXAITimeContext
{
   bool     ready;
   datetime server_now;
   datetime utc_now;
   datetime local_now;
   int      server_utc_offset_sec;
   int      local_utc_offset_sec;
   int      session_bucket;
   int      server_day_of_week;
   int      utc_day_of_week;
   int      local_day_of_week;
};

FXAITimeContext g_fxai_time_context;
bool            g_fxai_time_context_ready = false;

void FXAI_ResetTimeContext(FXAITimeContext &out)
{
   out.ready = false;
   out.server_now = 0;
   out.utc_now = 0;
   out.local_now = 0;
   out.server_utc_offset_sec = 0;
   out.local_utc_offset_sec = 0;
   out.session_bucket = 0;
   out.server_day_of_week = -1;
   out.utc_day_of_week = -1;
   out.local_day_of_week = -1;
}

datetime FXAI_RawServerNow(void)
{
   datetime now_time = TimeTradeServer();
   if(now_time > 0)
      return now_time;
   now_time = TimeCurrent();
   if(now_time > 0)
      return now_time;
   now_time = iTime(_Symbol, PERIOD_M1, 0);
   if(now_time > 0)
      return now_time;
   return TimeLocal();
}

datetime FXAI_RawUtcNow(void)
{
   datetime now_time = TimeGMT();
   if(now_time > 0)
      return now_time;
   if(g_fxai_time_context_ready && g_fxai_time_context.utc_now > 0)
      return g_fxai_time_context.utc_now;

   datetime server_now = FXAI_RawServerNow();
   if(server_now <= 0)
      return 0;

   int offset_sec = 0;
   if(g_fxai_time_context_ready)
      offset_sec = g_fxai_time_context.server_utc_offset_sec;
   return (server_now - offset_sec);
}

datetime FXAI_RawLocalNow(void)
{
   datetime now_time = TimeLocal();
   if(now_time > 0)
      return now_time;
   if(g_fxai_time_context_ready && g_fxai_time_context.local_now > 0)
      return g_fxai_time_context.local_now;
   return FXAI_RawServerNow();
}

bool FXAI_BuildTimeContext(FXAITimeContext &out)
{
   FXAI_ResetTimeContext(out);

   out.server_now = FXAI_RawServerNow();
   out.utc_now = FXAI_RawUtcNow();
   out.local_now = FXAI_RawLocalNow();
   if(out.server_now <= 0)
      return false;
   if(out.utc_now <= 0)
      out.utc_now = out.server_now;
   if(out.local_now <= 0)
      out.local_now = out.server_now;

   out.server_utc_offset_sec = (int)(out.server_now - out.utc_now);
   out.local_utc_offset_sec = (int)(out.local_now - out.utc_now);
   out.session_bucket = FXAI_DeriveSessionBucket(out.server_now);

   MqlDateTime dt;
   TimeToStruct(out.server_now, dt);
   out.server_day_of_week = dt.day_of_week;
   TimeToStruct(out.utc_now, dt);
   out.utc_day_of_week = dt.day_of_week;
   TimeToStruct(out.local_now, dt);
   out.local_day_of_week = dt.day_of_week;
   out.ready = true;
   return true;
}

void FXAI_ApplyTimeContext(const FXAITimeContext &ctx)
{
   g_fxai_time_context = ctx;
   g_fxai_time_context_ready = ctx.ready;
}

bool FXAI_RefreshTimeContext(void)
{
   FXAITimeContext ctx;
   if(!FXAI_BuildTimeContext(ctx))
      return false;
   FXAI_ApplyTimeContext(ctx);
   return true;
}

datetime FXAI_ServerNow(void)
{
   if(g_fxai_time_context_ready && g_fxai_time_context.server_now > 0)
      return g_fxai_time_context.server_now;
   return FXAI_RawServerNow();
}

datetime FXAI_UtcNow(void)
{
   if(g_fxai_time_context_ready && g_fxai_time_context.utc_now > 0)
      return g_fxai_time_context.utc_now;
   return FXAI_RawUtcNow();
}

datetime FXAI_LocalNow(void)
{
   if(g_fxai_time_context_ready && g_fxai_time_context.local_now > 0)
      return g_fxai_time_context.local_now;
   return FXAI_RawLocalNow();
}

datetime FXAI_ServerToUtc(const datetime server_time)
{
   if(server_time <= 0)
      return 0;
   int offset_sec = (g_fxai_time_context_ready ? g_fxai_time_context.server_utc_offset_sec : 0);
   return (server_time - offset_sec);
}

datetime FXAI_UtcToServer(const datetime utc_time)
{
   if(utc_time <= 0)
      return 0;
   int offset_sec = (g_fxai_time_context_ready ? g_fxai_time_context.server_utc_offset_sec : 0);
   return (utc_time + offset_sec);
}

datetime FXAI_LocalToServer(const datetime local_time)
{
   if(local_time <= 0)
      return 0;
   int local_offset = (g_fxai_time_context_ready ? g_fxai_time_context.local_utc_offset_sec : 0);
   int server_offset = (g_fxai_time_context_ready ? g_fxai_time_context.server_utc_offset_sec : 0);
   return (local_time - local_offset + server_offset);
}

string FXAI_TimeContextSummary(void)
{
   if(!g_fxai_time_context_ready)
      return "time_context_unavailable";
   return StringFormat("server=%d utc=%d local=%d server_offset=%d local_offset=%d session=%d",
                       (int)g_fxai_time_context.server_now,
                       (int)g_fxai_time_context.utc_now,
                       (int)g_fxai_time_context.local_now,
                       g_fxai_time_context.server_utc_offset_sec,
                       g_fxai_time_context.local_utc_offset_sec,
                       g_fxai_time_context.session_bucket);
}

#endif // __FXAI_RUNTIME_TIME_SERVICE_MQH__
