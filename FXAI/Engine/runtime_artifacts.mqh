#ifndef __FXAI_RUNTIME_ARTIFACTS_MQH__
#define __FXAI_RUNTIME_ARTIFACTS_MQH__

#define FXAI_RUNTIME_ARTIFACT_DIR "FXAI\\Runtime"
#define FXAI_RUNTIME_ARTIFACT_VERSION 14

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

string FXAI_RuntimeMacroManifestFile(const string symbol)
{
   return FXAI_RUNTIME_ARTIFACT_DIR + "\\fxai_macro_" + FXAI_RuntimeArtifactSafeSymbol(symbol) + ".tsv";
}

string FXAI_RuntimeShadowLedgerFile(const string symbol)
{
   return FXAI_RUNTIME_ARTIFACT_DIR + "\\fxai_shadow_" + FXAI_RuntimeArtifactSafeSymbol(symbol) + ".tsv";
}

bool FXAI_IsStatefulCheckpointManifest(const FXAIAIManifestV4 &manifest)
{
   return (FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_ONLINE_LEARNING) ||
           FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY) ||
           FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL));
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
   FileWriteDouble(handle, sample.point_value);
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
   sample.point_value = FileReadDouble(handle);
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

   FileWriteString(handle, "ai_id\tai_name\treference_tier\tcoverage_tag\tcheckpoint_depth\tpersistent\tstate_version\tcapability_mask\tstateful_checkpoint\tnative_snapshot\tdeterministic_replay\tnative_required\tpromotion_ready\tstate_file_size\tstate_file\tcoverage_note\r\n");
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
         bool stateful_checkpoint = FXAI_IsStatefulCheckpointManifest(manifest);
         bool native_required = stateful_checkpoint;
         string coverage_tag = plugin.PersistentStateCoverageTag();
         string depth_tag = plugin.PersistentStateDepthTag();
         bool has_native_snapshot = plugin.SupportsNativeParameterSnapshot();
         bool has_deterministic_replay = plugin.SupportsDeterministicReplayCheckpoint();
         bool promotion_ready = (!native_required || (coverage_tag == "native_model" && has_native_snapshot));
         string coverage_note = "";
         if(native_required && !promotion_ready)
            coverage_note = "stateful model blocked from live promotion until native parameter snapshot coverage is implemented";
         else if(coverage_tag == "native_model")
            coverage_note = "native checkpoint verified";
         else if(coverage_tag == "native_replay")
            coverage_note = "deterministic replay checkpoint available for audit and research recovery only";
         else
            coverage_note = "checkpoint not required";
         string line = IntegerToString(plugin.AIId()) + "\t" +
                       plugin.AIName() + "\t" +
                       FXAI_ReferenceTierName(manifest.reference_tier) + "\t" +
                       coverage_tag + "\t" +
                       depth_tag + "\t" +
                       IntegerToString(plugin.SupportsPersistentState() ? 1 : 0) + "\t" +
                       IntegerToString(plugin.PersistentStateVersion()) + "\t" +
                       IntegerToString((int)manifest.capability_mask) + "\t" +
                       IntegerToString(stateful_checkpoint ? 1 : 0) + "\t" +
                       IntegerToString(has_native_snapshot ? 1 : 0) + "\t" +
                       IntegerToString(has_deterministic_replay ? 1 : 0) + "\t" +
                       IntegerToString(native_required ? 1 : 0) + "\t" +
                       IntegerToString(promotion_ready ? 1 : 0) + "\t" +
                       IntegerToString((int)file_size) + "\t" +
                       state_file + "\t" +
                       coverage_note + "\r\n";
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

void FXAI_WriteMacroDatasetManifest(const string symbol)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_RUNTIME_ARTIFACT_DIR, FILE_COMMON);

   int handle = FileOpen(FXAI_RuntimeMacroManifestFile(symbol), FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;

   FXAIMacroEventDatasetStats stats;
   FXAI_GetMacroEventDatasetStats(stats);
   FileWriteString(handle, "symbol\tschema_version\trecord_count\tparse_errors\tdistinct_symbols\tdistinct_sources\tdistinct_event_ids\tdistinct_countries\tdistinct_currencies\tdistinct_revision_chains\tfamily_rates_count\tfamily_inflation_count\tfamily_labor_count\tfamily_growth_count\tfamily_trade_count\tfirst_event_time\tlast_event_time\tavg_importance\tavg_pre_window_min\tavg_post_window_min\tavg_surprise_z_abs\tavg_revision_abs\tavg_source_trust\tavg_currency_relevance\tchecksum01\tprovenance_hash01\tleakage_guard_score\tleakage_safe\r\n");
   string line = FXAI_RuntimeArtifactSafeSymbol(symbol) + "\t" +
                 IntegerToString(stats.schema_version) + "\t" +
                 IntegerToString(stats.record_count) + "\t" +
                 IntegerToString(stats.parse_errors) + "\t" +
                 IntegerToString(stats.distinct_symbols) + "\t" +
                 IntegerToString(stats.distinct_sources) + "\t" +
                 IntegerToString(stats.distinct_event_ids) + "\t" +
                 IntegerToString(stats.distinct_countries) + "\t" +
                 IntegerToString(stats.distinct_currencies) + "\t" +
                 IntegerToString(stats.distinct_revision_chains) + "\t" +
                 IntegerToString(stats.family_rates_count) + "\t" +
                 IntegerToString(stats.family_inflation_count) + "\t" +
                 IntegerToString(stats.family_labor_count) + "\t" +
                 IntegerToString(stats.family_growth_count) + "\t" +
                 IntegerToString(stats.family_trade_count) + "\t" +
                 IntegerToString((int)stats.first_event_time) + "\t" +
                 IntegerToString((int)stats.last_event_time) + "\t" +
                 DoubleToString(stats.avg_importance, 6) + "\t" +
                 DoubleToString(stats.avg_pre_window_min, 6) + "\t" +
                 DoubleToString(stats.avg_post_window_min, 6) + "\t" +
                 DoubleToString(stats.avg_surprise_z_abs, 6) + "\t" +
                 DoubleToString(stats.avg_revision_abs, 6) + "\t" +
                 DoubleToString(stats.avg_source_trust, 6) + "\t" +
                 DoubleToString(stats.avg_currency_relevance, 6) + "\t" +
                 DoubleToString(stats.checksum01, 6) + "\t" +
                 DoubleToString(stats.provenance_hash01, 6) + "\t" +
                 DoubleToString(stats.leakage_guard_score, 6) + "\t" +
                 IntegerToString(FXAI_MacroEventLeakageSafe() ? 1 : 0) + "\r\n";
   FileWriteString(handle, line);
   FileClose(handle);
}

void FXAI_WriteShadowFleetLedger(const string symbol)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_RUNTIME_ARTIFACT_DIR, FILE_COMMON);

   int handle = FileOpen(FXAI_RuntimeShadowLedgerFile(symbol),
                         FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return;

   FileWriteString(handle, "symbol\tai_id\tai_name\tfamily_id\tmeta_weight\treliability\tglobal_edge\tcontext_edge\tcontext_regret\tportfolio_objective\tportfolio_stability\tportfolio_corr\tportfolio_div\troute_value\troute_regret\troute_counterfactual\tshadow_score\tregime_id\thorizon_minutes\tobs\tpolicy_enter_prob\tpolicy_no_trade_prob\tpolicy_exit_prob\tpolicy_add_prob\tpolicy_reduce_prob\tpolicy_timeout_prob\tpolicy_tighten_prob\tpolicy_portfolio_fit\tpolicy_capital_efficiency\tpolicy_lifecycle_action\tportfolio_pressure\tcontrol_plane_score\tportfolio_supervisor_score\r\n");
   int r = g_ai_last_regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT)
      r = 0;
   int s = FXAI_DeriveSessionBucket(TimeCurrent());
   int hslot = FXAI_GetHorizonSlot(g_ai_last_horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS)
      hslot = 0;

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      CFXAIAIPlugin *plugin = (g_plugins_ready ? g_plugins.Get(ai) : NULL);
      if(plugin == NULL)
         continue;

      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);
      double meta = FXAI_Clamp(g_model_meta_weight[ai], 0.20, 3.00);
      double reliability = FXAI_Clamp(g_model_reliability[ai], 0.0, 1.0);
      double mm = MathMax(g_ai_last_min_move_points, 0.50);
      double global_edge = FXAI_Clamp(FXAI_GetModelRegimeEdge(ai, r) / mm, -4.0, 4.0) / 4.0;
      double context_edge = FXAI_Clamp(FXAI_GetModelContextEdge(ai, r, g_ai_last_horizon_minutes) / mm, -4.0, 4.0) / 4.0;
      double context_regret = FXAI_Clamp(FXAI_GetModelContextRegret(ai, r, g_ai_last_horizon_minutes), 0.0, 6.0) / 6.0;
      double port_obj = FXAI_Clamp(g_model_portfolio_objective[ai], -1.0, 1.0);
      double port_stab = FXAI_Clamp(g_model_portfolio_stability[ai], 0.0, 1.0);
      double port_corr = FXAI_Clamp(g_model_portfolio_corr_penalty[ai], 0.0, 1.0);
      double port_div = FXAI_Clamp(g_model_portfolio_diversification[ai], 0.0, 1.0);
      double route_value = FXAI_Clamp(g_model_plugin_route_value[ai][r][s][hslot], -1.0, 1.0);
      double route_regret = FXAI_Clamp(g_model_plugin_route_regret[ai][r][s][hslot], 0.0, 1.0);
      double route_cf = FXAI_Clamp(g_model_plugin_route_counterfactual[ai][r][s][hslot], -1.0, 1.0);
      int obs = g_model_plugin_route_obs[ai][r][s][hslot];
      double shadow_score = FXAI_Clamp(0.22 * reliability +
                                       0.16 * global_edge +
                                       0.16 * context_edge -
                                       0.14 * context_regret +
                                       0.14 * port_obj +
                                       0.12 * route_value +
                                       0.10 * route_cf -
                                       0.12 * route_regret +
                                       0.08 * port_stab -
                                       0.06 * port_corr +
                                       0.06 * port_div +
                                       0.05 * FXAI_Clamp((meta - 0.20) / 2.80, 0.0, 1.0),
                                       -1.0,
                                       1.0);
      string line = FXAI_RuntimeArtifactSafeSymbol(symbol) + "\t" +
                    IntegerToString(ai) + "\t" +
                    plugin.AIName() + "\t" +
                    IntegerToString(manifest.family) + "\t" +
                    DoubleToString(meta, 6) + "\t" +
                    DoubleToString(reliability, 6) + "\t" +
                    DoubleToString(global_edge, 6) + "\t" +
                    DoubleToString(context_edge, 6) + "\t" +
                    DoubleToString(context_regret, 6) + "\t" +
                    DoubleToString(port_obj, 6) + "\t" +
                    DoubleToString(port_stab, 6) + "\t" +
                    DoubleToString(port_corr, 6) + "\t" +
                    DoubleToString(port_div, 6) + "\t" +
                    DoubleToString(route_value, 6) + "\t" +
                    DoubleToString(route_regret, 6) + "\t" +
                    DoubleToString(route_cf, 6) + "\t" +
                    DoubleToString(shadow_score, 6) + "\t" +
                    IntegerToString(r) + "\t" +
                    IntegerToString(g_ai_last_horizon_minutes) + "\t" +
                    IntegerToString(obs) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_enter_prob, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_no_trade_prob, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_exit_prob, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_add_prob, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_reduce_prob, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_timeout_prob, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_tighten_prob, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_portfolio_fit, 0.0, 1.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_policy_last_capital_efficiency, 0.0, 1.0), 6) + "\t" +
                    IntegerToString(g_policy_last_action) + "\t" +
                    DoubleToString(FXAI_Clamp(g_ai_last_portfolio_pressure, 0.0, 2.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_control_plane_last_score, 0.0, 2.0), 6) + "\t" +
                    DoubleToString(FXAI_Clamp(g_portfolio_supervisor_last_score, 0.0, 3.0), 6) + "\r\n";
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

   FileWriteInteger(handle, (g_shared_transfer_global_ready ? 1 : 0));
   FileWriteInteger(handle, g_shared_transfer_global_steps);
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      FileWriteDouble(handle, g_shared_transfer_global_b[j]);
      FileWriteDouble(handle, g_shared_transfer_global_move[j]);
      for(int c=0; c<3; c++)
         FileWriteDouble(handle, g_shared_transfer_global_cls[c][j]);
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         FileWriteDouble(handle, g_shared_transfer_global_w[j][i]);
      for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
         FileWriteDouble(handle, g_shared_transfer_global_seq_w[j][t]);
      for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
         FileWriteDouble(handle, g_shared_transfer_global_time_w[j][c]);
      for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
         FileWriteDouble(handle, g_shared_transfer_global_time_gate_w[j][c]);
   }
   for(int d=0; d<FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS; d++)
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         FileWriteDouble(handle, g_shared_transfer_global_domain_emb[d][j]);
   for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         FileWriteDouble(handle, g_shared_transfer_global_horizon_emb[h][j]);
   for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         FileWriteDouble(handle, g_shared_transfer_global_session_emb[s][j]);
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      FileWriteDouble(handle, g_shared_transfer_global_state_rec_w[j]);
      FileWriteDouble(handle, g_shared_transfer_global_state_b[j]);
      for(int c=0; c<FXAI_SHARED_TRANSFER_STATE_FEATURES; c++)
         FileWriteDouble(handle, g_shared_transfer_global_state_w[j][c]);
   }

   FileWriteInteger(handle, (g_foundation_global_ready ? 1 : 0));
   FileWriteInteger(handle, g_foundation_global_steps);
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      FileWriteDouble(handle, g_foundation_mask_w[j]);
      FileWriteDouble(handle, g_foundation_vol_w[j]);
      FileWriteDouble(handle, g_foundation_shift_w[j]);
      FileWriteDouble(handle, g_foundation_ctx_w[j]);
   }
   FileWriteDouble(handle, g_foundation_mask_b);
   FileWriteDouble(handle, g_foundation_vol_b);
   FileWriteDouble(handle, g_foundation_shift_b);
   FileWriteDouble(handle, g_foundation_ctx_b);

   FileWriteInteger(handle, (g_student_global_ready ? 1 : 0));
   FileWriteInteger(handle, g_student_global_steps);
   for(int c=0; c<3; c++)
   {
      FileWriteDouble(handle, g_student_cls_b[c]);
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         FileWriteDouble(handle, g_student_cls[c][j]);
   }
   FileWriteDouble(handle, g_student_move_b);
   FileWriteDouble(handle, g_student_trade_b);
   FileWriteDouble(handle, g_student_horizon_b);
   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      FileWriteDouble(handle, g_student_move_w[j]);
      FileWriteDouble(handle, g_student_trade_w[j]);
      FileWriteDouble(handle, g_student_horizon_w[j]);
   }

   FileWriteInteger(handle, (g_analog_memory_ready ? 1 : 0));
   FileWriteInteger(handle, g_analog_memory_head);
   FileWriteInteger(handle, g_analog_memory_size);
   for(int i=0; i<FXAI_ANALOG_MEMORY_CAP; i++)
   {
      FileWriteLong(handle, (long)g_analog_memory_time[i]);
      FileWriteInteger(handle, g_analog_memory_regime[i]);
      FileWriteInteger(handle, g_analog_memory_session[i]);
      FileWriteInteger(handle, g_analog_memory_horizon[i]);
      FileWriteDouble(handle, g_analog_memory_domain_hash[i]);
      FileWriteDouble(handle, g_analog_memory_direction[i]);
      FileWriteDouble(handle, g_analog_memory_edge_norm[i]);
      FileWriteDouble(handle, g_analog_memory_quality[i]);
      FileWriteDouble(handle, g_analog_memory_path_risk[i]);
      FileWriteDouble(handle, g_analog_memory_fill_risk[i]);
      FileWriteDouble(handle, g_analog_memory_weight[i]);
      for(int j=0; j<FXAI_ANALOG_MEMORY_FEATS; j++)
         FileWriteDouble(handle, g_analog_memory_vec[i][j]);
   }

   FileWriteInteger(handle, (g_broker_execution_ready ? 1 : 0));
   for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
   {
      for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
      {
         FileWriteDouble(handle, g_broker_execution_obs[s][h]);
         FileWriteDouble(handle, g_broker_execution_slippage_ema[s][h]);
         FileWriteDouble(handle, g_broker_execution_latency_ema[s][h]);
         FileWriteDouble(handle, g_broker_execution_reject_ema[s][h]);
         FileWriteDouble(handle, g_broker_execution_partial_ema[s][h]);
      }
   }
   for(int idx=0; idx<FXAI_BROKER_EXEC_LIBRARY_CELLS; idx++)
   {
      FileWriteDouble(handle, g_broker_execution_library_obs[idx]);
      FileWriteDouble(handle, g_broker_execution_library_slippage[idx]);
      FileWriteDouble(handle, g_broker_execution_library_latency[idx]);
      FileWriteDouble(handle, g_broker_execution_library_reject[idx]);
      FileWriteDouble(handle, g_broker_execution_library_partial[idx]);
      FileWriteDouble(handle, g_broker_execution_library_fill_ratio[idx]);
   }
   for(int idx=0; idx<FXAI_BROKER_EXEC_LIBRARY_EVENT_CELLS; idx++)
      FileWriteDouble(handle, g_broker_execution_library_event_mass[idx]);
   FileWriteInteger(handle, g_broker_execution_trace_head);
   FileWriteInteger(handle, g_broker_execution_trace_size);
   for(int i=0; i<FXAI_BROKER_EXEC_TRACE_CAP; i++)
   {
      FileWriteLong(handle, (long)g_broker_execution_trace_time[i]);
      FileWriteInteger(handle, g_broker_execution_trace_symbol_bucket[i]);
      FileWriteInteger(handle, g_broker_execution_trace_session[i]);
      FileWriteInteger(handle, g_broker_execution_trace_horizon[i]);
      FileWriteInteger(handle, g_broker_execution_trace_side[i]);
      FileWriteInteger(handle, g_broker_execution_trace_order_type[i]);
      FileWriteInteger(handle, g_broker_execution_trace_event_kind[i]);
      FileWriteDouble(handle, g_broker_execution_trace_slippage[i]);
      FileWriteDouble(handle, g_broker_execution_trace_latency[i]);
      FileWriteDouble(handle, g_broker_execution_trace_reject[i]);
      FileWriteDouble(handle, g_broker_execution_trace_partial[i]);
      FileWriteDouble(handle, g_broker_execution_trace_fill_ratio[i]);
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
      FXAI_WriteMacroDatasetManifest(symbol);
      FXAI_WriteShadowFleetLedger(symbol);
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

         if(version >= 7)
         {
            g_shared_transfer_global_ready = (FileReadInteger(handle) != 0);
            g_shared_transfer_global_steps = FileReadInteger(handle);
            for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            {
               g_shared_transfer_global_b[j] = FileReadDouble(handle);
               g_shared_transfer_global_move[j] = FileReadDouble(handle);
               for(int c=0; c<3; c++)
                  g_shared_transfer_global_cls[c][j] = FileReadDouble(handle);
               for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
                  g_shared_transfer_global_w[j][i] = FileReadDouble(handle);
               if(version >= 9)
               {
                  for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
                     g_shared_transfer_global_seq_w[j][t] = FileReadDouble(handle);
               }
               if(version >= 11)
               {
                  for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
                     g_shared_transfer_global_time_w[j][c] = FileReadDouble(handle);
                  for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
                     g_shared_transfer_global_time_gate_w[j][c] = FileReadDouble(handle);
               }
            }
            for(int d=0; d<FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS; d++)
               for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
                  g_shared_transfer_global_domain_emb[d][j] = FileReadDouble(handle);
            for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
               for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
                  g_shared_transfer_global_horizon_emb[h][j] = FileReadDouble(handle);
            for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
               for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
                  g_shared_transfer_global_session_emb[s][j] = FileReadDouble(handle);
            if(version >= 12)
            {
               for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
               {
                  g_shared_transfer_global_state_rec_w[j] = FileReadDouble(handle);
                  g_shared_transfer_global_state_b[j] = FileReadDouble(handle);
                  for(int c=0; c<FXAI_SHARED_TRANSFER_STATE_FEATURES; c++)
                     g_shared_transfer_global_state_w[j][c] = FileReadDouble(handle);
               }
            }

            if(version >= 13)
            {
               g_foundation_global_ready = (FileReadInteger(handle) != 0);
               g_foundation_global_steps = FileReadInteger(handle);
               for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
               {
                  g_foundation_mask_w[j] = FileReadDouble(handle);
                  g_foundation_vol_w[j] = FileReadDouble(handle);
                  g_foundation_shift_w[j] = FileReadDouble(handle);
                  g_foundation_ctx_w[j] = FileReadDouble(handle);
               }
               g_foundation_mask_b = FileReadDouble(handle);
               g_foundation_vol_b = FileReadDouble(handle);
               g_foundation_shift_b = FileReadDouble(handle);
               g_foundation_ctx_b = FileReadDouble(handle);

               g_student_global_ready = (FileReadInteger(handle) != 0);
               g_student_global_steps = FileReadInteger(handle);
               for(int c=0; c<3; c++)
               {
                  g_student_cls_b[c] = FileReadDouble(handle);
                  for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
                     g_student_cls[c][j] = FileReadDouble(handle);
               }
               g_student_move_b = FileReadDouble(handle);
               g_student_trade_b = FileReadDouble(handle);
               g_student_horizon_b = FileReadDouble(handle);
               for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
               {
                  g_student_move_w[j] = FileReadDouble(handle);
                  g_student_trade_w[j] = FileReadDouble(handle);
                  g_student_horizon_w[j] = FileReadDouble(handle);
               }

               g_analog_memory_ready = (FileReadInteger(handle) != 0);
               g_analog_memory_head = FileReadInteger(handle);
               g_analog_memory_size = FileReadInteger(handle);
               for(int i=0; i<FXAI_ANALOG_MEMORY_CAP; i++)
               {
                  g_analog_memory_time[i] = (datetime)FileReadLong(handle);
                  g_analog_memory_regime[i] = FileReadInteger(handle);
                  g_analog_memory_session[i] = FileReadInteger(handle);
                  g_analog_memory_horizon[i] = FileReadInteger(handle);
                  g_analog_memory_domain_hash[i] = FileReadDouble(handle);
                  g_analog_memory_direction[i] = FileReadDouble(handle);
                  g_analog_memory_edge_norm[i] = FileReadDouble(handle);
                  g_analog_memory_quality[i] = FileReadDouble(handle);
                  g_analog_memory_path_risk[i] = FileReadDouble(handle);
                  g_analog_memory_fill_risk[i] = FileReadDouble(handle);
                  g_analog_memory_weight[i] = FileReadDouble(handle);
                  for(int j=0; j<FXAI_ANALOG_MEMORY_FEATS; j++)
                     g_analog_memory_vec[i][j] = FileReadDouble(handle);
               }
               if(g_analog_memory_size < 0)
                  g_analog_memory_size = 0;
               if(g_analog_memory_size > FXAI_ANALOG_MEMORY_CAP)
                  g_analog_memory_size = FXAI_ANALOG_MEMORY_CAP;
               if(g_analog_memory_head < 0 || g_analog_memory_head >= FXAI_ANALOG_MEMORY_CAP)
                  g_analog_memory_head = (g_analog_memory_size % FXAI_ANALOG_MEMORY_CAP);
               g_analog_memory_ready = (g_analog_memory_size >= FXAI_ANALOG_MEMORY_MIN_MATCHES);
            }

            g_broker_execution_ready = (FileReadInteger(handle) != 0);
            for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
            {
               for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
               {
                  g_broker_execution_obs[s][h] = FileReadDouble(handle);
                  g_broker_execution_slippage_ema[s][h] = FileReadDouble(handle);
                  g_broker_execution_latency_ema[s][h] = FileReadDouble(handle);
                  g_broker_execution_reject_ema[s][h] = FileReadDouble(handle);
                  g_broker_execution_partial_ema[s][h] = FileReadDouble(handle);
               }
            }
            if(version >= 12)
            {
               for(int idx=0; idx<FXAI_BROKER_EXEC_LIBRARY_CELLS; idx++)
               {
                  g_broker_execution_library_obs[idx] = FileReadDouble(handle);
                  g_broker_execution_library_slippage[idx] = FileReadDouble(handle);
                  g_broker_execution_library_latency[idx] = FileReadDouble(handle);
                  g_broker_execution_library_reject[idx] = FileReadDouble(handle);
                  g_broker_execution_library_partial[idx] = FileReadDouble(handle);
                  g_broker_execution_library_fill_ratio[idx] = FileReadDouble(handle);
               }
               for(int idx=0; idx<FXAI_BROKER_EXEC_LIBRARY_EVENT_CELLS; idx++)
                  g_broker_execution_library_event_mass[idx] = FileReadDouble(handle);
            }
            if(version >= 10)
            {
               g_broker_execution_trace_head = FileReadInteger(handle);
               g_broker_execution_trace_size = FileReadInteger(handle);
               for(int i=0; i<FXAI_BROKER_EXEC_TRACE_CAP; i++)
               {
                  g_broker_execution_trace_time[i] = (datetime)FileReadLong(handle);
                  if(version >= 11)
                     g_broker_execution_trace_symbol_bucket[i] = FileReadInteger(handle);
                  g_broker_execution_trace_session[i] = FileReadInteger(handle);
                  g_broker_execution_trace_horizon[i] = FileReadInteger(handle);
                  if(version >= 11)
                  {
                     g_broker_execution_trace_side[i] = FileReadInteger(handle);
                     g_broker_execution_trace_order_type[i] = FileReadInteger(handle);
                     g_broker_execution_trace_event_kind[i] = FileReadInteger(handle);
                  }
                  g_broker_execution_trace_slippage[i] = FileReadDouble(handle);
                  g_broker_execution_trace_latency[i] = FileReadDouble(handle);
                  g_broker_execution_trace_reject[i] = FileReadDouble(handle);
                  g_broker_execution_trace_partial[i] = FileReadDouble(handle);
                  if(version >= 11)
                     g_broker_execution_trace_fill_ratio[i] = FileReadDouble(handle);
               }
            }
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
