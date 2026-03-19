#ifndef __FXAI_RUNTIME_ARTIFACTS_MQH__
#define __FXAI_RUNTIME_ARTIFACTS_MQH__

#define FXAI_RUNTIME_ARTIFACT_DIR "FXAI\\Runtime"
#define FXAI_RUNTIME_ARTIFACT_VERSION 3

string FXAI_RuntimeArtifactSafeSymbol(const string symbol)
{
   string clean = symbol;
   if(StringLen(clean) <= 0)
      clean = _Symbol;
   StringReplace(clean, "\\", "_");
   StringReplace(clean, "/", "_");
   StringReplace(clean, ":", "_");
   StringReplace(clean, "*", "_");
   StringReplace(clean, "?", "_");
   StringReplace(clean, "\"", "_");
   StringReplace(clean, "<", "_");
   StringReplace(clean, ">", "_");
   StringReplace(clean, "|", "_");
   return clean;
}

string FXAI_RuntimeArtifactFile(const string symbol)
{
   return FXAI_RUNTIME_ARTIFACT_DIR + "\\fxai_runtime_" + FXAI_RuntimeArtifactSafeSymbol(symbol) + ".bin";
}

string FXAI_RuntimePersistenceManifestFile(const string symbol)
{
   return FXAI_RUNTIME_ARTIFACT_DIR + "\\fxai_persistence_" + FXAI_RuntimeArtifactSafeSymbol(symbol) + ".tsv";
}

string FXAI_RuntimeFeatureManifestFile(const string symbol)
{
   return FXAI_RUNTIME_ARTIFACT_DIR + "\\fxai_features_" + FXAI_RuntimeArtifactSafeSymbol(symbol) + ".tsv";
}

void FXAI_MarkRuntimeArtifactsDirty(void)
{
   g_runtime_artifacts_dirty = true;
}

long FXAI_CommonFileSize(const string file_name)
{
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return -1;
   long sz = (long)FileSize(handle);
   FileClose(handle);
   return sz;
}

void FXAI_WritePreparedSample(const int handle,
                              const FXAIPreparedSample &sample)
{
   FileWriteInteger(handle, (sample.valid ? 1 : 0));
   FileWriteInteger(handle, sample.label_class);
   FileWriteInteger(handle, sample.regime_id);
   FileWriteInteger(handle, sample.horizon_minutes);
   FileWriteInteger(handle, sample.horizon_slot);
   FileWriteDouble(handle, sample.move_points);
   FileWriteDouble(handle, sample.min_move_points);
   FileWriteDouble(handle, sample.cost_points);
   FileWriteDouble(handle, sample.sample_weight);
   FileWriteDouble(handle, sample.quality_score);
   FileWriteDouble(handle, sample.mfe_points);
   FileWriteDouble(handle, sample.mae_points);
   FileWriteDouble(handle, sample.spread_stress);
   FileWriteDouble(handle, sample.trace_spread_mean_ratio);
   FileWriteDouble(handle, sample.trace_spread_peak_ratio);
   FileWriteDouble(handle, sample.trace_range_mean_ratio);
   FileWriteDouble(handle, sample.trace_body_efficiency);
   FileWriteDouble(handle, sample.trace_gap_ratio);
   FileWriteDouble(handle, sample.trace_reversal_ratio);
   FileWriteDouble(handle, sample.trace_session_transition);
   FileWriteDouble(handle, sample.trace_rollover);
   FileWriteDouble(handle, sample.time_to_hit_frac);
   FileWriteInteger(handle, sample.path_flags);
   FileWriteDouble(handle, sample.masked_step_target);
   FileWriteDouble(handle, sample.next_vol_target);
   FileWriteDouble(handle, sample.regime_shift_target);
   FileWriteDouble(handle, sample.context_lead_target);
   FileWriteDouble(handle, sample.domain_hash);
   FileWriteLong(handle, (long)sample.sample_time);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      FileWriteDouble(handle, sample.x[k]);
}

void FXAI_ReadPreparedSample(const int handle,
                             FXAIPreparedSample &sample)
{
   sample.valid = (FileReadInteger(handle) != 0);
   sample.label_class = FileReadInteger(handle);
   sample.regime_id = FileReadInteger(handle);
   sample.horizon_minutes = FileReadInteger(handle);
   sample.horizon_slot = FileReadInteger(handle);
   sample.move_points = FileReadDouble(handle);
   sample.min_move_points = FileReadDouble(handle);
   sample.cost_points = FileReadDouble(handle);
   sample.sample_weight = FileReadDouble(handle);
   sample.quality_score = FileReadDouble(handle);
   sample.mfe_points = FileReadDouble(handle);
   sample.mae_points = FileReadDouble(handle);
   sample.spread_stress = FileReadDouble(handle);
   sample.trace_spread_mean_ratio = FileReadDouble(handle);
   sample.trace_spread_peak_ratio = FileReadDouble(handle);
   sample.trace_range_mean_ratio = FileReadDouble(handle);
   sample.trace_body_efficiency = FileReadDouble(handle);
   sample.trace_gap_ratio = FileReadDouble(handle);
   sample.trace_reversal_ratio = FileReadDouble(handle);
   sample.trace_session_transition = FileReadDouble(handle);
   sample.trace_rollover = FileReadDouble(handle);
   sample.time_to_hit_frac = FileReadDouble(handle);
   sample.path_flags = FileReadInteger(handle);
   sample.masked_step_target = FileReadDouble(handle);
   sample.next_vol_target = FileReadDouble(handle);
   sample.regime_shift_target = FileReadDouble(handle);
   sample.context_lead_target = FileReadDouble(handle);
   sample.domain_hash = FileReadDouble(handle);
   sample.sample_time = (datetime)FileReadLong(handle);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      sample.x[k] = FileReadDouble(handle);
}

void FXAI_WritePersistenceCoverageManifest(const string symbol)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_RUNTIME_ARTIFACT_DIR, FILE_COMMON);

   int handle = FileOpen(FXAI_RuntimePersistenceManifestFile(symbol), FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;

   FileWriteString(handle, "ai_id\tai_name\treference_tier\tcoverage_tag\tpersistent\tstate_version\tcapability_mask\tstate_file_size\tstate_file\r\n");
   if(g_plugins_ready)
   {
      for(int ai=0; ai<FXAI_AI_COUNT; ai++)
      {
         CFXAIAIPlugin *plugin = g_plugins.Get(ai);
         if(plugin == NULL)
            continue;

         FXAIAIManifestV4 manifest;
         FXAI_GetPluginManifest(*plugin, manifest);
         string state_file = plugin.PersistentStateFile(symbol);
         long file_size = FXAI_CommonFileSize(state_file);
         string line = IntegerToString(plugin.AIId()) + "\t" +
                       plugin.AIName() + "\t" +
                       FXAI_ReferenceTierName(manifest.reference_tier) + "\t" +
                       plugin.PersistentStateCoverageTag() + "\t" +
                       IntegerToString(plugin.SupportsPersistentState() ? 1 : 0) + "\t" +
                       IntegerToString(plugin.PersistentStateVersion()) + "\t" +
                       IntegerToString((int)manifest.capability_mask) + "\t" +
                       IntegerToString((int)file_size) + "\t" +
                       state_file + "\r\n";
         FileWriteString(handle, line);
      }
   }

   FileClose(handle);
}

void FXAI_WriteFeatureRegistryManifest(const string symbol)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_RUNTIME_ARTIFACT_DIR, FILE_COMMON);

   int handle = FileOpen(FXAI_RuntimeFeatureManifestFile(symbol), FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;

   FileWriteString(handle, "feature_idx\tfeature_name\tfeature_group\tprovenance\tleakage_guarded\tclip_lo\tclip_hi\r\n");
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double lo = -8.0;
      double hi = 8.0;
      FXAI_GetFeatureClipBounds(f, lo, hi);
      string line = IntegerToString(f) + "\t" +
                    FXAI_FeatureName(f) + "\t" +
                    FXAI_FeatureGroupName(FXAI_GetFeatureGroupForIndex(f)) + "\t" +
                    FXAI_FeatureProvenanceName(FXAI_FeatureProvenance(f)) + "\t" +
                    IntegerToString(FXAI_FeatureLeakageGuarded(f) ? 1 : 0) + "\t" +
                    DoubleToString(lo, 6) + "\t" +
                    DoubleToString(hi, 6) + "\r\n";
      FileWriteString(handle, line);
   }

   FileClose(handle);
}

bool FXAI_SaveRuntimeArtifacts(const string symbol)
{
   string file_name = FXAI_RuntimeArtifactFile(symbol);
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_RUNTIME_ARTIFACT_DIR, FILE_COMMON);
   int handle = FileOpen(file_name, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   FileWriteInteger(handle, FXAI_RUNTIME_ARTIFACT_VERSION);
   FileWriteInteger(handle, FXAI_AI_FEATURES);
   FileWriteInteger(handle, FXAI_NORM_METHOD_COUNT);
   FileWriteInteger(handle, FXAI_NORM_ROLL_WINDOW_MAX);
   FileWriteInteger(handle, FXAI_REPLAY_CAPACITY);
   FileWriteInteger(handle, FXAI_AI_COUNT);
   FileWriteInteger(handle, FXAI_REGIME_COUNT);
   FileWriteInteger(handle, FXAI_MAX_HORIZONS);
   FileWriteInteger(handle, FXAI_CONFORMAL_DEPTH);
   FileWriteInteger(handle, FXAI_REL_MAX_PENDING);

   FileWriteInteger(handle, (g_norm_windows_ready ? 1 : 0));
   FileWriteInteger(handle, g_norm_default_window);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      FileWriteInteger(handle, g_norm_feature_windows[f]);

   FileWriteInteger(handle, (g_fxai_norm_window_inited ? 1 : 0));
   FileWriteInteger(handle, g_fxai_norm_default_window);
   FileWriteInteger(handle, g_fxai_norm_window_cfg_version);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      FileWriteInteger(handle, g_fxai_norm_feature_window[f]);

   FileWriteInteger(handle, (g_fxai_norm_hist_inited ? 1 : 0));
   for(int m=0; m<FXAI_NORM_METHOD_COUNT; m++)
   {
      FileWriteLong(handle, (long)g_fxai_norm_last_sample_time[m]);
      FileWriteInteger(handle, g_fxai_norm_last_cfg_version[m]);
      for(int f=0; f<FXAI_AI_FEATURES; f++)
      {
         FileWriteInteger(handle, g_fxai_norm_hist_count[m][f]);
         FileWriteInteger(handle, g_fxai_norm_hist_head[m][f]);
         for(int k=0; k<FXAI_NORM_ROLL_WINDOW_MAX; k++)
            FileWriteDouble(handle, g_fxai_norm_hist[m][f][k]);
      }
   }

   FileWriteInteger(handle, g_replay_count);
   FileWriteInteger(handle, g_replay_cursor);
   for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      FileWriteLong(handle, (long)g_replay_last_sample_time[h]);
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         FileWriteInteger(handle, g_replay_bucket_count[r][h]);
   for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
   {
      FileWriteInteger(handle, (g_replay_used[i] ? 1 : 0));
      FileWriteDouble(handle, g_replay_priority[i]);
      FileWriteInteger(handle, g_replay_flags[i]);
      FXAI_WritePreparedSample(handle, g_replay_samples[i]);
   }

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            FileWriteInteger(handle, g_conf_count[ai][r][h]);
            FileWriteInteger(handle, g_conf_head[ai][r][h]);
            for(int i=0; i<FXAI_CONFORMAL_DEPTH; i++)
            {
               FileWriteDouble(handle, g_conf_class_score[ai][r][h][i]);
               FileWriteDouble(handle, g_conf_move_score[ai][r][h][i]);
               FileWriteDouble(handle, g_conf_path_score[ai][r][h][i]);
            }
         }
      }

      FileWriteInteger(handle, g_conf_pending_head[ai]);
      FileWriteInteger(handle, g_conf_pending_tail[ai]);
      for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
      {
         FileWriteInteger(handle, g_conf_pending_seq[ai][k]);
         FileWriteInteger(handle, g_conf_pending_regime[ai][k]);
         FileWriteInteger(handle, g_conf_pending_horizon[ai][k]);
         FileWriteDouble(handle, g_conf_pending_prob[ai][k][0]);
         FileWriteDouble(handle, g_conf_pending_prob[ai][k][1]);
         FileWriteDouble(handle, g_conf_pending_prob[ai][k][2]);
         FileWriteDouble(handle, g_conf_pending_move_q25[ai][k]);
         FileWriteDouble(handle, g_conf_pending_move_q50[ai][k]);
         FileWriteDouble(handle, g_conf_pending_move_q75[ai][k]);
         FileWriteDouble(handle, g_conf_pending_path_risk[ai][k]);
      }
   }

   FileWriteInteger(handle, (g_feature_drift_ready ? 1 : 0));
   FileWriteLong(handle, (long)g_feature_drift_last_time);
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      FileWriteInteger(handle, g_feature_drift_baseline_obs[g]);
      FileWriteInteger(handle, g_feature_drift_live_obs[g]);
      FileWriteDouble(handle, g_feature_drift_baseline_mean[g]);
      FileWriteDouble(handle, g_feature_drift_baseline_abs[g]);
      FileWriteDouble(handle, g_feature_drift_live_mean[g]);
      FileWriteDouble(handle, g_feature_drift_live_abs[g]);
      FileWriteDouble(handle, g_feature_drift_ema[g]);
   }

   FileClose(handle);

   bool ok = true;
   if(g_plugins_ready)
   {
      for(int ai=0; ai<FXAI_AI_COUNT; ai++)
      {
         CFXAIAIPlugin *plugin = g_plugins.Get(ai);
         if(plugin == NULL || !plugin.SupportsPersistentState())
            continue;
         if(!plugin.SaveStateFile(plugin.PersistentStateFile(symbol)))
            ok = false;
      }
   }

   if(ok)
   {
      FXAI_WritePersistenceCoverageManifest(symbol);
      FXAI_WriteFeatureRegistryManifest(symbol);
      g_runtime_artifacts_dirty = false;
      g_runtime_last_save_time = TimeCurrent();
   }
   return ok;
}

bool FXAI_LoadRuntimeArtifacts(const string symbol)
{
   bool loaded_global = false;
   string file_name = FXAI_RuntimeArtifactFile(symbol);
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      int version = FileReadInteger(handle);
      int features = FileReadInteger(handle);
      int norm_methods = FileReadInteger(handle);
      int norm_window_max = FileReadInteger(handle);
      int replay_cap = FileReadInteger(handle);
      int ai_count = FileReadInteger(handle);
      int regimes = FileReadInteger(handle);
      int horizons = FileReadInteger(handle);
      int conformal_depth = FileReadInteger(handle);
      int pending_cap = FileReadInteger(handle);
      bool ok = (version == FXAI_RUNTIME_ARTIFACT_VERSION &&
                 features == FXAI_AI_FEATURES &&
                 norm_methods == FXAI_NORM_METHOD_COUNT &&
                 norm_window_max == FXAI_NORM_ROLL_WINDOW_MAX &&
                 replay_cap == FXAI_REPLAY_CAPACITY &&
                 ai_count == FXAI_AI_COUNT &&
                 regimes == FXAI_REGIME_COUNT &&
                 horizons == FXAI_MAX_HORIZONS &&
                 conformal_depth == FXAI_CONFORMAL_DEPTH &&
                 pending_cap == FXAI_REL_MAX_PENDING);
      if(ok)
      {
         g_norm_windows_ready = (FileReadInteger(handle) != 0);
         g_norm_default_window = FileReadInteger(handle);
         for(int f=0; f<FXAI_AI_FEATURES; f++)
            g_norm_feature_windows[f] = FileReadInteger(handle);

         g_fxai_norm_window_inited = (FileReadInteger(handle) != 0);
         g_fxai_norm_default_window = FileReadInteger(handle);
         g_fxai_norm_window_cfg_version = FileReadInteger(handle);
         for(int f=0; f<FXAI_AI_FEATURES; f++)
            g_fxai_norm_feature_window[f] = FileReadInteger(handle);

         g_fxai_norm_hist_inited = (FileReadInteger(handle) != 0);
         for(int m=0; m<FXAI_NORM_METHOD_COUNT; m++)
         {
            g_fxai_norm_last_sample_time[m] = (datetime)FileReadLong(handle);
            g_fxai_norm_last_cfg_version[m] = FileReadInteger(handle);
            for(int f=0; f<FXAI_AI_FEATURES; f++)
            {
               g_fxai_norm_hist_count[m][f] = FileReadInteger(handle);
               g_fxai_norm_hist_head[m][f] = FileReadInteger(handle);
               for(int k=0; k<FXAI_NORM_ROLL_WINDOW_MAX; k++)
                  g_fxai_norm_hist[m][f][k] = FileReadDouble(handle);
            }
         }

         g_replay_count = FileReadInteger(handle);
         g_replay_cursor = FileReadInteger(handle);
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
            g_replay_last_sample_time[h] = (datetime)FileReadLong(handle);
         for(int r=0; r<FXAI_REGIME_COUNT; r++)
            for(int h=0; h<FXAI_MAX_HORIZONS; h++)
               g_replay_bucket_count[r][h] = FileReadInteger(handle);
         for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
         {
            g_replay_used[i] = (FileReadInteger(handle) != 0);
            g_replay_priority[i] = FileReadDouble(handle);
            g_replay_flags[i] = FileReadInteger(handle);
            FXAI_ReadPreparedSample(handle, g_replay_samples[i]);
         }

         for(int ai=0; ai<FXAI_AI_COUNT; ai++)
         {
            for(int r=0; r<FXAI_REGIME_COUNT; r++)
            {
               for(int h=0; h<FXAI_MAX_HORIZONS; h++)
               {
                  g_conf_count[ai][r][h] = FileReadInteger(handle);
                  g_conf_head[ai][r][h] = FileReadInteger(handle);
                  for(int i=0; i<FXAI_CONFORMAL_DEPTH; i++)
                  {
                     g_conf_class_score[ai][r][h][i] = FileReadDouble(handle);
                     g_conf_move_score[ai][r][h][i] = FileReadDouble(handle);
                     g_conf_path_score[ai][r][h][i] = FileReadDouble(handle);
                  }
               }
            }

            g_conf_pending_head[ai] = FileReadInteger(handle);
            g_conf_pending_tail[ai] = FileReadInteger(handle);
            for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
            {
               g_conf_pending_seq[ai][k] = FileReadInteger(handle);
               g_conf_pending_regime[ai][k] = FileReadInteger(handle);
               g_conf_pending_horizon[ai][k] = FileReadInteger(handle);
               g_conf_pending_prob[ai][k][0] = FileReadDouble(handle);
               g_conf_pending_prob[ai][k][1] = FileReadDouble(handle);
               g_conf_pending_prob[ai][k][2] = FileReadDouble(handle);
               g_conf_pending_move_q25[ai][k] = FileReadDouble(handle);
               g_conf_pending_move_q50[ai][k] = FileReadDouble(handle);
               g_conf_pending_move_q75[ai][k] = FileReadDouble(handle);
               g_conf_pending_path_risk[ai][k] = FileReadDouble(handle);
            }
         }

         g_feature_drift_ready = (FileReadInteger(handle) != 0);
         g_feature_drift_last_time = (datetime)FileReadLong(handle);
         for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
         {
            g_feature_drift_baseline_obs[g] = FileReadInteger(handle);
            g_feature_drift_live_obs[g] = FileReadInteger(handle);
            g_feature_drift_baseline_mean[g] = FileReadDouble(handle);
            g_feature_drift_baseline_abs[g] = FileReadDouble(handle);
            g_feature_drift_live_mean[g] = FileReadDouble(handle);
            g_feature_drift_live_abs[g] = FileReadDouble(handle);
            g_feature_drift_ema[g] = FileReadDouble(handle);
         }

         loaded_global = true;
      }
      FileClose(handle);
   }

   bool loaded_plugin = false;
   if(g_plugins_ready)
   {
      for(int ai=0; ai<FXAI_AI_COUNT; ai++)
      {
         CFXAIAIPlugin *plugin = g_plugins.Get(ai);
         if(plugin == NULL || !plugin.SupportsPersistentState())
            continue;
         if(plugin.LoadStateFile(plugin.PersistentStateFile(symbol)))
            loaded_plugin = true;
      }
   }

   if(loaded_global || loaded_plugin)
   {
      g_runtime_artifacts_dirty = false;
      g_runtime_last_save_time = TimeCurrent();
   }
   return (loaded_global || loaded_plugin);
}

void FXAI_MaybeSaveRuntimeArtifacts(const string symbol,
                                    const datetime bar_time)
{
   if(!g_runtime_artifacts_dirty)
      return;
   datetime now = bar_time;
   if(now <= 0)
      now = TimeCurrent();
   if(g_runtime_last_save_time > 0 && (now - g_runtime_last_save_time) < 900)
      return;
   FXAI_SaveRuntimeArtifacts(symbol);
}

#endif // __FXAI_RUNTIME_ARTIFACTS_MQH__
