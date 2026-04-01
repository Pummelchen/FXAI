#property strict

input string Export_OutputKey = "default";
input long   Export_WindowStartUnix = 0;
input long   Export_WindowEndUnix = 0;
input int    Export_MaxBars = 600000;
input bool   Export_ResetOutput = true;
input int    TradeKiller = 0;

#define FXAI_OFFLINE_EXPORT_DIR "FXAI\\Offline\\Exports"

string FXAI_OfflineSafeToken(const string raw)
{
   string out = raw;
   if(StringLen(out) <= 0)
      out = "default";
   StringReplace(out, "\\", "_");
   StringReplace(out, "/", "_");
   StringReplace(out, ":", "_");
   StringReplace(out, "*", "_");
   StringReplace(out, "?", "_");
   StringReplace(out, "\"", "_");
   StringReplace(out, "<", "_");
   StringReplace(out, ">", "_");
   StringReplace(out, "|", "_");
   StringReplace(out, " ", "_");
   return out;
}

string FXAI_OfflineExportStem(void)
{
   return "fxai_export_" + FXAI_OfflineSafeToken(Export_OutputKey) + "_" + FXAI_OfflineSafeToken(_Symbol);
}

string FXAI_OfflineExportDataFile(void)
{
   return FXAI_OFFLINE_EXPORT_DIR + "\\" + FXAI_OfflineExportStem() + ".tsv";
}

string FXAI_OfflineExportMetaFile(void)
{
   return FXAI_OFFLINE_EXPORT_DIR + "\\" + FXAI_OfflineExportStem() + ".meta.tsv";
}

bool FXAI_WriteOfflineMeta(const int bars_written,
                           const datetime first_time,
                           const datetime last_time)
{
   int handle = FileOpen(FXAI_OfflineExportMetaFile(),
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   FileWriteString(handle, "key\tvalue\r\n");
   FileWriteString(handle, "output_key\t" + FXAI_OfflineSafeToken(Export_OutputKey) + "\r\n");
   FileWriteString(handle, "symbol\t" + _Symbol + "\r\n");
   FileWriteString(handle, "timeframe\tM1\r\n");
   FileWriteString(handle, "window_start_unix\t" + IntegerToString((int)Export_WindowStartUnix) + "\r\n");
   FileWriteString(handle, "window_end_unix\t" + IntegerToString((int)Export_WindowEndUnix) + "\r\n");
   FileWriteString(handle, "bars_written\t" + IntegerToString(bars_written) + "\r\n");
   FileWriteString(handle, "first_time_unix\t" + IntegerToString((int)first_time) + "\r\n");
   FileWriteString(handle, "last_time_unix\t" + IntegerToString((int)last_time) + "\r\n");
   FileWriteString(handle, "first_time_text\t" + TimeToString(first_time, TIME_DATE | TIME_MINUTES) + "\r\n");
   FileWriteString(handle, "last_time_text\t" + TimeToString(last_time, TIME_DATE | TIME_MINUTES) + "\r\n");
   FileClose(handle);
   return true;
}

bool FXAI_RunOfflineExport(void)
{
   datetime start_time = (Export_WindowStartUnix > 0 ? (datetime)Export_WindowStartUnix : 0);
   datetime end_time = (Export_WindowEndUnix > 0 ? (datetime)Export_WindowEndUnix : 0);
   if(start_time <= 0 || end_time <= start_time)
   {
      Print("FXAI offline export: invalid window start=", (long)start_time,
            " end=", (long)end_time);
      return false;
   }

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int got = CopyRates(_Symbol, PERIOD_M1, start_time, end_time, rates);
   if(got <= 0)
   {
      Print("FXAI offline export: CopyRates failed symbol=", _Symbol,
            " start=", (long)start_time,
            " end=", (long)end_time);
      return false;
   }

   int max_bars = Export_MaxBars;
   if(max_bars < 1)
      max_bars = got;
   if(max_bars > got)
      max_bars = got;

   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate("FXAI\\Offline", FILE_COMMON);
   FolderCreate(FXAI_OFFLINE_EXPORT_DIR, FILE_COMMON);

   string data_file = FXAI_OfflineExportDataFile();
   string meta_file = FXAI_OfflineExportMetaFile();
   if(Export_ResetOutput)
   {
      FileDelete(data_file, FILE_COMMON);
      FileDelete(meta_file, FILE_COMMON);
   }

   int handle = FileOpen(data_file,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      Print("FXAI offline export: failed to open data file ", data_file);
      return false;
   }

   FileWriteString(handle, "time_unix\topen\thigh\tlow\tclose\tspread_points\ttick_volume\treal_volume\r\n");
   datetime first_time = 0;
   datetime last_time = 0;
   int oldest = max_bars - 1;
   for(int i=oldest; i>=0; i--)
   {
      datetime t = rates[i].time;
      if(first_time <= 0)
         first_time = t;
      last_time = t;
      string line = IntegerToString((int)t) + "\t" +
                    DoubleToString(rates[i].open, _Digits) + "\t" +
                    DoubleToString(rates[i].high, _Digits) + "\t" +
                    DoubleToString(rates[i].low, _Digits) + "\t" +
                    DoubleToString(rates[i].close, _Digits) + "\t" +
                    IntegerToString((int)rates[i].spread) + "\t" +
                    IntegerToString((int)rates[i].tick_volume) + "\t" +
                    IntegerToString((int)rates[i].real_volume) + "\r\n";
      FileWriteString(handle, line);
   }
   FileClose(handle);

   if(!FXAI_WriteOfflineMeta(max_bars, first_time, last_time))
   {
      Print("FXAI offline export: failed to write meta file ", meta_file);
      return false;
   }

   Print("FXAI offline export done: symbol=", _Symbol,
         " bars=", max_bars,
         " start=", TimeToString(first_time, TIME_DATE | TIME_MINUTES),
         " end=", TimeToString(last_time, TIME_DATE | TIME_MINUTES),
         " file=FILE_COMMON/", data_file);
   return true;
}

int OnInit()
{
   bool ok = FXAI_RunOfflineExport();
   if(MQLInfoInteger(MQL_TESTER))
      TesterStop();
   else
      ExpertRemove();
   return (ok ? INIT_SUCCEEDED : INIT_FAILED);
}

void OnTick()
{
}
