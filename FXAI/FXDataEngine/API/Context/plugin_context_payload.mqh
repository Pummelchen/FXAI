   double InputCostProxyPoints(const double &x[]) const
   {
      if(m_ctx_cost_ready && m_ctx_cost_points >= 0.0)
         return m_ctx_cost_points;

      // Fallback when context has not been set explicitly before a dry-run call.
      if(ArraySize(x) <= 7) return 0.0;
      return MathMax(0.0, MathAbs(x[7]));
   }

   double ResolveCostPoints(const double &x[]) const
   {
      return InputCostProxyPoints(x);
   }

   double ResolveMinMovePoints(void) const
   {
      if(m_ctx_min_move_points > 0.0) return m_ctx_min_move_points;
      return 0.0;
   }

   double ResolvePointValue(void) const
   {
      if(MathIsValidNumber(m_ctx_point_value) && m_ctx_point_value > 0.0)
         return m_ctx_point_value;
      return (_Point > 0.0 ? _Point : 1.0);
   }

   datetime ResolveContextTime(void) const
   {
      if(m_ctx_time_ready && m_ctx_time > 0) return m_ctx_time;
      return TimeCurrent();
   }

   void SetContext(const FXAIAIContextV4 &ctx)
   {
      m_ctx_time_ready = (ctx.sample_time > 0);
      m_ctx_time = (m_ctx_time_ready ? ctx.sample_time : 0);

      m_ctx_cost_ready = (MathIsValidNumber(ctx.cost_points) && ctx.cost_points >= 0.0);
      m_ctx_cost_points = (m_ctx_cost_ready ? ctx.cost_points : 0.0);

      if(MathIsValidNumber(ctx.min_move_points) && ctx.min_move_points > 0.0)
         m_ctx_min_move_points = ctx.min_move_points;
      else
         m_ctx_min_move_points = 0.0;

      if(ctx.regime_id >= 0 && ctx.regime_id < FXAI_PLUGIN_REGIME_BUCKETS)
         m_ctx_regime_id = ctx.regime_id;
      else
         m_ctx_regime_id = 0;

      if(ctx.session_bucket >= 0 && ctx.session_bucket < FXAI_PLUGIN_SESSION_BUCKETS)
         m_ctx_session_bucket = ctx.session_bucket;
      else
         m_ctx_session_bucket = FXAI_DeriveSessionBucket(ctx.sample_time);

      m_ctx_horizon_minutes = (ctx.horizon_minutes > 0 ? ctx.horizon_minutes : 1);
      m_ctx_feature_schema_id = ctx.feature_schema_id;
      if(m_ctx_feature_schema_id < FXAI_SCHEMA_FULL || m_ctx_feature_schema_id > FXAI_SCHEMA_CONTEXTUAL)
         m_ctx_feature_schema_id = FXAI_SCHEMA_FULL;
      m_ctx_normalization_method_id = ctx.normalization_method_id;
      if(m_ctx_normalization_method_id < 0 || m_ctx_normalization_method_id >= FXAI_NORM_METHOD_COUNT)
         m_ctx_normalization_method_id = FXAI_NORM_EXISTING;
      m_ctx_sequence_bars = (ctx.sequence_bars > 0 ? ctx.sequence_bars : 1);
      if(m_ctx_sequence_bars > FXAI_MAX_SEQUENCE_BARS)
         m_ctx_sequence_bars = FXAI_MAX_SEQUENCE_BARS;
      m_ctx_point_value = (MathIsValidNumber(ctx.point_value) && ctx.point_value > 0.0
                           ? ctx.point_value : (_Point > 0.0 ? _Point : 1.0));
      m_ctx_domain_hash = (MathIsValidNumber(ctx.domain_hash) && ctx.domain_hash >= 0.0 && ctx.domain_hash <= 1.0
                           ? ctx.domain_hash : FXAI_SymbolHash01(_Symbol));
   }

   void SetWindowPayload(const int window_size,
                         const double &x_window[][FXAI_AI_WEIGHTS])
   {
      m_ctx_window_size = window_size;
      if(m_ctx_window_size < 0) m_ctx_window_size = 0;
      if(m_ctx_window_size > FXAI_MAX_SEQUENCE_BARS) m_ctx_window_size = FXAI_MAX_SEQUENCE_BARS;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = (b < m_ctx_window_size ? x_window[b][k] : 0.0);
      }
   }

   void ClearWindowPayload(void)
   {
      m_ctx_window_size = 0;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = 0.0;
   }

   void BuildSharedAdapterInput(const double &x[],
                                double &out[]) const
   {
      FXAI_BuildSharedTransferInputGlobal(x,
                                         m_ctx_window,
                                         m_ctx_window_size,
                                         m_ctx_domain_hash,
                                         m_ctx_horizon_minutes,
                                         out);
   }

   bool HasSharedAdapterSignal(const double &a[]) const
   {
      if(ArraySize(a) < FXAI_SHARED_TRANSFER_FEATURES) return false;
      double mag = 0.0;
      for(int i=1; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         mag += MathAbs(a[i]);
      return (mag > 1e-6);
   }

   double SharedAdapterSignalStrength(const double &a[]) const
   {
      if(ArraySize(a) < FXAI_SHARED_TRANSFER_FEATURES)
         return 0.0;
      return FXAI_Clamp(0.14 +
                        0.07 * a[4] +
                        0.04 * MathAbs(a[8]) +
                        0.04 * MathAbs(a[11]) +
                        0.04 * MathAbs(a[12]) +
                        0.04 * MathAbs(a[15]) +
                        0.03 * MathAbs(a[18]) -
                        0.02 * MathAbs(a[19]) +
                        0.05 * MathAbs(a[20]) +
                        0.05 * MathAbs(a[21]) +
                        0.04 * a[22] +
                        0.04 * MathAbs(a[23]) +
                        0.04 * MathAbs(a[24]) +
                        0.03 * MathAbs(a[25]) +
                        0.03 * MathAbs(a[26]) -
                        0.03 * a[27],
                        0.0,
                        0.62);
   }

