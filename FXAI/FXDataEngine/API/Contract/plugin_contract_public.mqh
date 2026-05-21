   CFXAIAIPlugin(void) { ResetAuxState(); }

   virtual int AIId(void) const = 0;
   virtual string AIName(void) const = 0;

   virtual void Reset(void)
   {
      ResetAuxState();
      m_native_quality_heads.Reset();
   }
   virtual bool SupportsPersistentState(void) const { return true; }
   virtual int PersistentStateVersion(void) const { return 12; }
   virtual bool SupportsDeterministicReplayCheckpoint(void) const { return true; }
   virtual bool SupportsNativeParameterSnapshot(void) const { return false; }
   virtual string PersistentStateDepthTag(void) const
   {
      FXAIAIManifestV4 manifest;
      DescribeResolved(manifest);
      bool stateful = (FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_ONLINE_LEARNING) ||
                       FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY) ||
                       FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL));
      if(!stateful)
         return "stateless";
      if(SupportsNativeParameterSnapshot())
         return "native_parameters";
      if(SupportsDeterministicReplayCheckpoint())
         return "deterministic_replay";
      return "base_only";
   }
   virtual string PersistentStateCoverageTag(void) const
   {
      FXAIAIManifestV4 manifest;
      DescribeResolved(manifest);
      if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_ONLINE_LEARNING) ||
         FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY) ||
         FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL))
      {
         return (SupportsNativeParameterSnapshot() ? "native_model" : "native_replay");
      }
      return FXAI_ReferenceTierName(FXAI_DefaultReferenceTierForAI(AIId()));
   }
   virtual void Describe(FXAIAIManifestV4 &out) const = 0;
   void DescribeResolved(FXAIAIManifestV4 &out) const
   {
      Describe(out);
      if(out.feature_groups_mask == 0)
         out.feature_groups_mask = FXAI_DefaultFeatureGroupsForFamily(out.family);
      if(out.feature_schema_id <= 0)
         out.feature_schema_id = FXAI_DefaultFeatureSchemaForFamily(out.family);
      if(out.reference_tier < (int)FXAI_REFERENCE_FULL_NATIVE ||
         out.reference_tier > (int)FXAI_REFERENCE_RULE_BASELINE)
      {
         out.reference_tier = FXAI_DefaultReferenceTierForAI(AIId());
      }
   }
   virtual bool SupportsSyntheticSeries(void) const { return false; }
   virtual bool SetSyntheticSeries(const datetime &time_arr[],
                                   const double &open_arr[],
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[])
   {
      return false;
   }
   virtual void ClearSyntheticSeries(void) {}
   virtual void ResetState(const int reason, const datetime when)
   {
      Reset();
   }
   virtual bool SelfTest(void)
   {
      FXAIAIManifestV4 manifest;
      DescribeResolved(manifest);
      return (manifest.api_version == FXAI_API_VERSION_V4 &&
              manifest.ai_id == AIId() &&
              StringLen(manifest.ai_name) > 0);
   }
   virtual void EnsureInitialized(const FXAIAIHyperParams &hp) {}

   int CorePredictFailures(void) const { return m_core_predict_failures; }
   int ReplayRehearsals(void) const { return m_replay_rehearsals; }

   string PersistentStateFile(const string symbol) const
   {
      string clean_symbol = symbol;
      if(StringLen(clean_symbol) <= 0)
         clean_symbol = _Symbol;
      StringReplace(clean_symbol, "\\", "_");
      StringReplace(clean_symbol, "/", "_");
      StringReplace(clean_symbol, ":", "_");
      StringReplace(clean_symbol, "*", "_");
      StringReplace(clean_symbol, "?", "_");
      StringReplace(clean_symbol, "\"", "_");
      StringReplace(clean_symbol, "<", "_");
      StringReplace(clean_symbol, ">", "_");
      StringReplace(clean_symbol, "|", "_");

      string clean_name = AIName();
      StringReplace(clean_name, "\\", "_");
      StringReplace(clean_name, "/", "_");
      StringReplace(clean_name, ":", "_");
      StringReplace(clean_name, "*", "_");
      StringReplace(clean_name, "?", "_");
      StringReplace(clean_name, "\"", "_");
      StringReplace(clean_name, "<", "_");
      StringReplace(clean_name, ">", "_");
      StringReplace(clean_name, "|", "_");
      return FXAI_PLUGIN_STATE_ARTIFACT_DIR + "\\fxai_plugin_" + clean_symbol + "_" + clean_name + ".bin";
   }

   bool SaveStateFile(const string file_name) const
   {
      FolderCreate("FXAI", FILE_COMMON);
      FolderCreate("FXAI\\Runtime", FILE_COMMON);
      FolderCreate(FXAI_PLUGIN_STATE_ARTIFACT_DIR, FILE_COMMON);
      int handle = FileOpen(file_name, FILE_WRITE | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE)
         return false;

      FileWriteInteger(handle, FXAI_PLUGIN_STATE_ARTIFACT_VERSION);
      FileWriteInteger(handle, AIId());
      FileWriteInteger(handle, PersistentStateVersion());
      FileWriteInteger(handle, (SupportsPersistentState() ? 1 : 0));
      bool ok = true;
      if(SupportsPersistentState())
      {
         ok = SaveBasePersistentState(handle);
         if(ok)
            ok = SaveModelState(handle);
      }
      FileClose(handle);
      return ok;
   }

   bool LoadStateFile(const string file_name)
   {
      int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE)
         return false;

      int version = FileReadInteger(handle);
      int ai_id = FileReadInteger(handle);
      int model_version = FileReadInteger(handle);
      bool persisted = (FileReadInteger(handle) != 0);
      bool ok = (version == FXAI_PLUGIN_STATE_ARTIFACT_VERSION &&
                 ai_id == AIId() &&
                 persisted &&
                 SupportsPersistentState());
      if(ok)
      {
         Reset();
         ok = LoadBasePersistentState(handle, model_version);
         if(ok)
            ok = LoadModelState(handle, model_version);
      }
      FileClose(handle);
      return ok;
   }

   void Train(const FXAIAITrainRequestV4 &req, const FXAIAIHyperParams &hp)
   {
      string reason = "";
      if(!FXAI_ValidateTrainRequestV4(req, reason))
         return;

      FXAIAIManifestV4 manifest;
      DescribeResolved(manifest);
      if(!FXAI_ValidateManifestContextCompatibilityV4(manifest, req.ctx, reason))
         return;
      EnsureInitialized(hp);
      SetContext(req.ctx);
      SetWindowPayload(req.window_size, req.x_window);
      SetTrainingTargets(req);
      bool can_learn = FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_ONLINE_LEARNING);
      bool can_replay = can_learn && FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY);
      if(!can_learn)
         return;

      m_persist_hp = hp;
      m_persist_hp_ready = true;
      if(m_persist_train_events < 1000000)
         m_persist_train_events++;

      double pre_probs[3];
      pre_probs[0] = 0.10;
      pre_probs[1] = 0.10;
      pre_probs[2] = 0.80;
      double pre_move = 0.0;
      bool have_pre = PredictModelCore(req.x, hp, pre_probs, pre_move);
      if(!have_pre)
         m_core_predict_failures++;
      NormalizeClassDistribution(pre_probs);
      if(!MathIsValidNumber(pre_move) || pre_move < 0.0)
         pre_move = 0.0;

      double sample_w = (req.sample_weight > 0.0 ? req.sample_weight : MoveSampleWeight(req.x, req.move_points));
      if(have_pre)
         UpdateContextCalibrationBank(req.label_class, pre_probs, pre_move, req.move_points, sample_w);
      double replay_pri = ComputeReplayPriority(req.label_class,
                                                pre_probs,
                                                req.move_points,
                                                req.ctx.cost_points,
                                                req.ctx.min_move_points);
      if(can_replay)
         StoreReplaySample(req, replay_pri);

      UpdateCrossSymbolTransferBank(req.x, req.move_points, sample_w);
      TrainModelCore(req.label_class, req.x, hp, req.move_points);
      UpdateSharedContextAdapter(req.x, req.label_class, req.move_points, sample_w, hp.lr);
      UpdateQualityHeads(req, sample_w);
      if(can_replay)
         RunReplayRehearsal(hp, req.ctx.regime_id, req.ctx.horizon_minutes);
      FXAI_MarkRuntimeArtifactsDirty();
   }

   void TrainSharedTransfer(const FXAIAIContextV4 &ctx,
                            const double &x[],
                            const double &x_window[][FXAI_AI_WEIGHTS],
                            const int window_size,
                            const double move_points,
                            const double sample_w,
                            const double lr)
   {
      string reason = "";
      if(!FXAI_ValidateContextV4(ctx, reason))
         return;
      if(!FXAI_ValidateWindowPayloadV4(x_window, window_size, reason))
         return;
      if(window_size > MathMax(ctx.sequence_bars - 1, 0))
         return;
      if(ctx.sequence_bars > 1 && window_size <= 0)
         return;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         if(!MathIsValidNumber(x[k]))
            return;
      }
      SetContext(ctx);
      SetWindowPayload(window_size, x_window);
      UpdateCrossSymbolTransferBank(x, move_points, sample_w);
      UpdateSharedContextAdapter(x,
                                 NormalizeClassLabel(-1, x, move_points),
                                 move_points,
                                 sample_w,
                                 FXAI_Clamp(0.50 * lr, 0.0001, 0.0500));
      FXAI_MarkRuntimeArtifactsDirty();
   }

   bool Predict(const FXAIAIPredictRequestV4 &req,
                const FXAIAIHyperParams &hp,
                FXAIAIPredictionV4 &out)
   {
      string reason = "";
      if(!FXAI_ValidatePredictRequestV4(req, reason))
      {
         for(int c=0; c<3; c++)
            out.class_probs[c] = (c == (int)FXAI_LABEL_SKIP ? 1.0 : 0.0);
         out.move_mean_points = 0.0;
         out.move_q25_points = 0.0;
         out.move_q50_points = 0.0;
         out.move_q75_points = 0.0;
         out.mfe_mean_points = 0.0;
         out.mae_mean_points = 0.0;
         out.hit_time_frac = 1.0;
         out.path_risk = 0.0;
         out.fill_risk = 0.0;
         out.confidence = 0.0;
         out.reliability = 0.0;
         return false;
      }
      FXAIAIManifestV4 manifest;
      DescribeResolved(manifest);
      FXAIAIModelOutputV4 model_out;
      if(!FXAI_ValidateManifestContextCompatibilityV4(manifest, req.ctx, reason))
      {
         out.class_probs[(int)FXAI_LABEL_SELL] = 0.0;
         out.class_probs[(int)FXAI_LABEL_BUY] = 0.0;
         out.class_probs[(int)FXAI_LABEL_SKIP] = 1.0;
         out.move_mean_points = 0.0;
         out.move_q25_points = 0.0;
         out.move_q50_points = 0.0;
         out.move_q75_points = 0.0;
         out.mfe_mean_points = 0.0;
         out.mae_mean_points = 0.0;
         out.hit_time_frac = 1.0;
         out.path_risk = 0.0;
         out.fill_risk = 0.0;
         out.confidence = 0.0;
         out.reliability = 0.0;
         return false;
      }
      EnsureInitialized(hp);
      SetContext(req.ctx);
      SetWindowPayload(req.window_size, req.x_window);
      ResetModelOutput(model_out);
      m_core_predict_calls++;
      if(!PredictDistributionCore(req.x, hp, model_out))
      {
         m_core_predict_failures++;
         ResetModelOutput(model_out);
         model_out.class_probs[(int)FXAI_LABEL_SELL] = 0.05;
         model_out.class_probs[(int)FXAI_LABEL_BUY] = 0.05;
         model_out.class_probs[(int)FXAI_LABEL_SKIP] = 0.90;
         model_out.move_mean_points = 0.0;
         FillPredictionV4(model_out, model_out.move_mean_points, out);
         return false;
      }

      NormalizeClassDistribution(model_out.class_probs);
      ApplySharedContextAdapter(model_out, req.x);
      if(!model_out.has_path_quality)
      {
         double structural = (model_out.has_confidence ? model_out.reliability : 0.50);
         double execution = (model_out.has_confidence ? model_out.confidence : 0.50);
         PopulatePathQualityHeads(model_out, req.x, FXAI_Clamp(1.0 - model_out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0), structural, execution);
      }
      ApplyContextCalibrationBank(model_out.class_probs);
      double calibrated_move = ApplyExpectedMoveCalibrationBank(model_out.move_mean_points);
      FillPredictionV4(model_out, calibrated_move, out);
      FXAI_ApplyConformalPredictionAdjustment(AIId(),
                                              req.ctx.regime_id,
                                              req.ctx.horizon_minutes,
                                              req.ctx.min_move_points,
                                              out);
      return true;
   }


