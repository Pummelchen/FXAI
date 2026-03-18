#ifndef __FXAI_DATA_NORM_WINDOWS_MQH__
#define __FXAI_DATA_NORM_WINDOWS_MQH__

#include "core.mqh"

#define FXAI_NORM_ROLL_WINDOW_DEFAULT 192
#define FXAI_NORM_ROLL_WINDOW_MAX 512

int g_fxai_norm_default_window = FXAI_NORM_ROLL_WINDOW_DEFAULT;
int g_fxai_norm_feature_window[FXAI_AI_FEATURES];
int g_fxai_norm_window_cfg_version = 0;
bool g_fxai_norm_window_inited = false;

int FXAI_NormalizationWindowClamp(const int w)
{
   int v = w;
   if(v < 16) v = 16;
   if(v > FXAI_NORM_ROLL_WINDOW_MAX) v = FXAI_NORM_ROLL_WINDOW_MAX;
   return v;
}

void FXAI_ResetNormalizationWindows(const int default_window = FXAI_NORM_ROLL_WINDOW_DEFAULT)
{
   g_fxai_norm_default_window = FXAI_NormalizationWindowClamp(default_window);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      g_fxai_norm_feature_window[f] = g_fxai_norm_default_window;
   g_fxai_norm_window_inited = true;
   g_fxai_norm_window_cfg_version++;
   FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_SetNormalizationWindows(const int &windows[], const int default_window = FXAI_NORM_ROLL_WINDOW_DEFAULT)
{
   int def_w = FXAI_NormalizationWindowClamp(default_window);
   if(!g_fxai_norm_window_inited)
      FXAI_ResetNormalizationWindows(def_w);
   else
      g_fxai_norm_default_window = def_w;

   int n = ArraySize(windows);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int w = def_w;
      if(f < n) w = FXAI_NormalizationWindowClamp(windows[f]);
      g_fxai_norm_feature_window[f] = w;
   }
   g_fxai_norm_window_cfg_version++;
   FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_GetNormalizationWindows(int &out_windows[], int &out_default_window)
{
   if(!g_fxai_norm_window_inited)
      FXAI_ResetNormalizationWindows(FXAI_NORM_ROLL_WINDOW_DEFAULT);

   if(ArraySize(out_windows) != FXAI_AI_FEATURES)
      ArrayResize(out_windows, FXAI_AI_FEATURES);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      out_windows[f] = g_fxai_norm_feature_window[f];
   out_default_window = g_fxai_norm_default_window;
}


#endif // __FXAI_DATA_NORM_WINDOWS_MQH__
