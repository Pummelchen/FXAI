#ifndef __FXAI_META_PERSISTENCE_MQH__
#define __FXAI_META_PERSISTENCE_MQH__

#define FXAI_META_ARTIFACT_DIR "FXAI\\Meta"
#define FXAI_META_ARTIFACT_VERSION 6

string FXAI_MetaArtifactFile(const string symbol)
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
   return FXAI_META_ARTIFACT_DIR + "\\fxai_meta_" + clean + ".bin";
}

void FXAI_MarkMetaArtifactsDirty(void)
{
   g_meta_artifacts_dirty = true;
}

bool FXAI_SaveMetaArtifacts(const string symbol)
{
   string file_name = FXAI_MetaArtifactFile(symbol);
   FolderCreate(FXAI_META_ARTIFACT_DIR, FILE_COMMON);
   int handle = FileOpen(file_name, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   FileWriteInteger(handle, FXAI_META_ARTIFACT_VERSION);
   FileWriteInteger(handle, FXAI_REGIME_COUNT);
   FileWriteInteger(handle, FXAI_STACK_FEATS);
   FileWriteInteger(handle, FXAI_STACK_HIDDEN);
   FileWriteInteger(handle, FXAI_TRADE_GATE_FEATS);
   FileWriteInteger(handle, FXAI_TRADE_GATE_HIDDEN);
   FileWriteInteger(handle, FXAI_HPOL_FEATS);
   FileWriteInteger(handle, FXAI_HPOL_HIDDEN);
   FileWriteInteger(handle, FXAI_AI_COUNT);

   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      FileWriteInteger(handle, (g_stack_ready[r] ? 1 : 0));
      FileWriteInteger(handle, g_stack_obs[r]);
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         FileWriteDouble(handle, g_stack_b1[r][h]);
         for(int k=0; k<FXAI_STACK_FEATS; k++)
            FileWriteDouble(handle, g_stack_w1[r][h][k]);
      }
      for(int c=0; c<3; c++)
      {
         FileWriteDouble(handle, g_stack_b2[r][c]);
         for(int h=0; h<FXAI_STACK_HIDDEN; h++)
            FileWriteDouble(handle, g_stack_w2[r][c][h]);
      }
      for(int slot=0; slot<FXAI_MAX_HORIZONS; slot++)
      {
         for(int c=0; c<3; c++)
         {
            FileWriteDouble(handle, g_router_action_value[r][slot][c]);
            FileWriteDouble(handle, g_router_action_regret[r][slot][c]);
            FileWriteDouble(handle, g_router_action_counterfactual[r][slot][c]);
            FileWriteInteger(handle, (g_router_action_ready[r][slot][c] ? 1 : 0));
            FileWriteInteger(handle, g_router_action_obs[r][slot][c]);
         }
      }

      FileWriteInteger(handle, (g_trade_gate_ready[r] ? 1 : 0));
      FileWriteInteger(handle, g_trade_gate_obs[r]);
      for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
      {
         FileWriteDouble(handle, g_trade_gate_b1[r][h]);
         for(int k=0; k<FXAI_TRADE_GATE_FEATS; k++)
            FileWriteDouble(handle, g_trade_gate_w1[r][h][k]);
      }
      FileWriteDouble(handle, g_trade_gate_b2[r]);
      for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
         FileWriteDouble(handle, g_trade_gate_w2[r][h]);

      FileWriteInteger(handle, (g_hpolicy_ready[r] ? 1 : 0));
      FileWriteInteger(handle, g_hpolicy_obs[r]);
      for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
      {
         FileWriteDouble(handle, g_hpolicy_b1[r][h]);
         for(int k=0; k<FXAI_HPOL_FEATS; k++)
            FileWriteDouble(handle, g_hpolicy_w1[r][h][k]);
      }
      FileWriteDouble(handle, g_hpolicy_b2[r]);
      for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
         FileWriteDouble(handle, g_hpolicy_w2[r][h]);

      for(int slot=0; slot<FXAI_MAX_HORIZONS; slot++)
      {
         FileWriteInteger(handle, (g_meta_oof_ready[r][slot] ? 1 : 0));
         FileWriteInteger(handle, g_meta_oof_obs[r][slot]);
         FileWriteDouble(handle, g_meta_oof_score_ema[r][slot]);
         FileWriteDouble(handle, g_meta_oof_edge_ema[r][slot]);
         FileWriteDouble(handle, g_meta_oof_quality_ema[r][slot]);
         FileWriteDouble(handle, g_meta_oof_trade_rate_ema[r][slot]);
      }
   }

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      FileWriteDouble(handle, g_model_meta_weight[ai]);
      FileWriteDouble(handle, g_model_global_edge_ema[ai]);
      FileWriteInteger(handle, (g_model_global_edge_ready[ai] ? 1 : 0));
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         FileWriteDouble(handle, g_model_regime_edge_ema[ai][r]);
         FileWriteInteger(handle, (g_model_regime_edge_ready[ai][r] ? 1 : 0));
         FileWriteInteger(handle, g_model_regime_obs[ai][r]);
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            FileWriteDouble(handle, g_model_context_edge_ema[ai][r][h]);
            FileWriteDouble(handle, g_model_context_regret_ema[ai][r][h]);
            FileWriteInteger(handle, (g_model_context_ready[ai][r][h] ? 1 : 0));
            FileWriteInteger(handle, g_model_context_obs[ai][r][h]);
         }
      }
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         FileWriteDouble(handle, g_model_horizon_edge_ema[ai][h]);
         FileWriteInteger(handle, (g_model_horizon_edge_ready[ai][h] ? 1 : 0));
         FileWriteInteger(handle, g_model_horizon_obs[ai][h]);
      }
      FileWriteDouble(handle, g_model_portfolio_mean_edge[ai]);
      FileWriteDouble(handle, g_model_portfolio_stability[ai]);
      FileWriteDouble(handle, g_model_portfolio_corr_penalty[ai]);
      FileWriteDouble(handle, g_model_portfolio_diversification[ai]);
      FileWriteDouble(handle, g_model_portfolio_objective[ai]);
      FileWriteInteger(handle, (g_model_portfolio_ready[ai] ? 1 : 0));
      FileWriteInteger(handle, g_model_portfolio_symbol_count[ai]);
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
         {
            for(int h=0; h<FXAI_MAX_HORIZONS; h++)
            {
               FileWriteDouble(handle, g_model_plugin_route_value[ai][r][s][h]);
               FileWriteDouble(handle, g_model_plugin_route_regret[ai][r][s][h]);
               FileWriteDouble(handle, g_model_plugin_route_counterfactual[ai][r][s][h]);
               FileWriteInteger(handle, (g_model_plugin_route_ready[ai][r][s][h] ? 1 : 0));
               FileWriteInteger(handle, g_model_plugin_route_obs[ai][r][s][h]);
            }
         }
      }
   }

   FileClose(handle);
   g_meta_artifacts_dirty = false;
   g_meta_last_save_time = TimeCurrent();
   return true;
}

bool FXAI_LoadMetaArtifacts(const string symbol)
{
   string file_name = FXAI_MetaArtifactFile(symbol);
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   bool ok = true;
   int version = FileReadInteger(handle);
   int regimes = FileReadInteger(handle);
   int stack_feats = FileReadInteger(handle);
   int stack_hidden = FileReadInteger(handle);
   int gate_feats = FileReadInteger(handle);
   int gate_hidden = FileReadInteger(handle);
   int hpol_feats = FileReadInteger(handle);
   int hpol_hidden = FileReadInteger(handle);
   int ai_count = FileReadInteger(handle);
   if(version != FXAI_META_ARTIFACT_VERSION ||
      regimes != FXAI_REGIME_COUNT ||
      stack_feats != FXAI_STACK_FEATS ||
      stack_hidden != FXAI_STACK_HIDDEN ||
      gate_feats != FXAI_TRADE_GATE_FEATS ||
      gate_hidden != FXAI_TRADE_GATE_HIDDEN ||
      hpol_feats != FXAI_HPOL_FEATS ||
      hpol_hidden != FXAI_HPOL_HIDDEN ||
      ai_count != FXAI_AI_COUNT)
   {
      ok = false;
   }

   if(ok)
   {
      for(int r=0; r<FXAI_REGIME_COUNT && ok; r++)
      {
         g_stack_ready[r] = (FileReadInteger(handle) != 0);
         g_stack_obs[r] = FileReadInteger(handle);
         for(int h=0; h<FXAI_STACK_HIDDEN && ok; h++)
         {
            g_stack_b1[r][h] = FileReadDouble(handle);
            for(int k=0; k<FXAI_STACK_FEATS; k++)
               g_stack_w1[r][h][k] = FileReadDouble(handle);
         }
         for(int c=0; c<3 && ok; c++)
         {
            g_stack_b2[r][c] = FileReadDouble(handle);
            for(int h=0; h<FXAI_STACK_HIDDEN; h++)
               g_stack_w2[r][c][h] = FileReadDouble(handle);
         }
         if(version >= 5)
         {
            for(int slot=0; slot<FXAI_MAX_HORIZONS; slot++)
            {
               for(int c=0; c<3; c++)
               {
                  g_router_action_value[r][slot][c] = FileReadDouble(handle);
                  g_router_action_regret[r][slot][c] = FileReadDouble(handle);
                  g_router_action_counterfactual[r][slot][c] = FileReadDouble(handle);
                  g_router_action_ready[r][slot][c] = (FileReadInteger(handle) != 0);
                  g_router_action_obs[r][slot][c] = FileReadInteger(handle);
               }
            }
         }

         g_trade_gate_ready[r] = (FileReadInteger(handle) != 0);
         g_trade_gate_obs[r] = FileReadInteger(handle);
         for(int h=0; h<FXAI_TRADE_GATE_HIDDEN && ok; h++)
         {
            g_trade_gate_b1[r][h] = FileReadDouble(handle);
            for(int k=0; k<FXAI_TRADE_GATE_FEATS; k++)
               g_trade_gate_w1[r][h][k] = FileReadDouble(handle);
         }
         g_trade_gate_b2[r] = FileReadDouble(handle);
         for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
            g_trade_gate_w2[r][h] = FileReadDouble(handle);

         g_hpolicy_ready[r] = (FileReadInteger(handle) != 0);
         g_hpolicy_obs[r] = FileReadInteger(handle);
         for(int h=0; h<FXAI_HPOL_HIDDEN && ok; h++)
         {
            g_hpolicy_b1[r][h] = FileReadDouble(handle);
            for(int k=0; k<FXAI_HPOL_FEATS; k++)
               g_hpolicy_w1[r][h][k] = FileReadDouble(handle);
         }
         g_hpolicy_b2[r] = FileReadDouble(handle);
         for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
            g_hpolicy_w2[r][h] = FileReadDouble(handle);

         for(int slot=0; slot<FXAI_MAX_HORIZONS; slot++)
         {
            g_meta_oof_ready[r][slot] = (FileReadInteger(handle) != 0);
            g_meta_oof_obs[r][slot] = FileReadInteger(handle);
            g_meta_oof_score_ema[r][slot] = FileReadDouble(handle);
            g_meta_oof_edge_ema[r][slot] = FileReadDouble(handle);
            g_meta_oof_quality_ema[r][slot] = FileReadDouble(handle);
            g_meta_oof_trade_rate_ema[r][slot] = FileReadDouble(handle);
         }
      }

      for(int ai=0; ai<FXAI_AI_COUNT && ok; ai++)
      {
         g_model_meta_weight[ai] = FileReadDouble(handle);
         g_model_global_edge_ema[ai] = FileReadDouble(handle);
         g_model_global_edge_ready[ai] = (FileReadInteger(handle) != 0);
         for(int r=0; r<FXAI_REGIME_COUNT; r++)
         {
            g_model_regime_edge_ema[ai][r] = FileReadDouble(handle);
            g_model_regime_edge_ready[ai][r] = (FileReadInteger(handle) != 0);
            g_model_regime_obs[ai][r] = FileReadInteger(handle);
            for(int h=0; h<FXAI_MAX_HORIZONS; h++)
            {
               g_model_context_edge_ema[ai][r][h] = FileReadDouble(handle);
               g_model_context_regret_ema[ai][r][h] = FileReadDouble(handle);
               g_model_context_ready[ai][r][h] = (FileReadInteger(handle) != 0);
               g_model_context_obs[ai][r][h] = FileReadInteger(handle);
            }
         }
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            g_model_horizon_edge_ema[ai][h] = FileReadDouble(handle);
            g_model_horizon_edge_ready[ai][h] = (FileReadInteger(handle) != 0);
            g_model_horizon_obs[ai][h] = FileReadInteger(handle);
         }
         g_model_portfolio_mean_edge[ai] = FileReadDouble(handle);
         g_model_portfolio_stability[ai] = FileReadDouble(handle);
         g_model_portfolio_corr_penalty[ai] = FileReadDouble(handle);
         g_model_portfolio_diversification[ai] = FileReadDouble(handle);
         double stored_portfolio_objective = 0.0;
         if(version >= 6)
            stored_portfolio_objective = FileReadDouble(handle);
         g_model_portfolio_ready[ai] = (FileReadInteger(handle) != 0);
         g_model_portfolio_symbol_count[ai] = FileReadInteger(handle);
         if(version >= 6)
            g_model_portfolio_objective[ai] = stored_portfolio_objective;
         else
            g_model_portfolio_objective[ai] = FXAI_ComputePortfolioObjective(g_model_portfolio_mean_edge[ai],
                                                                             g_model_portfolio_stability[ai],
                                                                             g_model_portfolio_corr_penalty[ai],
                                                                             g_model_portfolio_diversification[ai],
                                                                             g_model_portfolio_symbol_count[ai]);
         if(version >= 6)
         {
            for(int r=0; r<FXAI_REGIME_COUNT; r++)
            {
               for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
               {
                  for(int h=0; h<FXAI_MAX_HORIZONS; h++)
                  {
                     g_model_plugin_route_value[ai][r][s][h] = FileReadDouble(handle);
                     g_model_plugin_route_regret[ai][r][s][h] = FileReadDouble(handle);
                     g_model_plugin_route_counterfactual[ai][r][s][h] = FileReadDouble(handle);
                     g_model_plugin_route_ready[ai][r][s][h] = (FileReadInteger(handle) != 0);
                     g_model_plugin_route_obs[ai][r][s][h] = FileReadInteger(handle);
                  }
               }
            }
         }
      }
   }

   FileClose(handle);
   if(!ok)
      return false;

   g_meta_artifacts_dirty = false;
   g_meta_last_save_time = TimeCurrent();
   return true;
}

void FXAI_MaybeSaveMetaArtifacts(const string symbol,
                                 const datetime bar_time)
{
   if(!g_meta_artifacts_dirty)
      return;
   datetime now = bar_time;
   if(now <= 0)
      now = TimeCurrent();
   if(g_meta_last_save_time > 0 && (now - g_meta_last_save_time) < 900)
      return;
   FXAI_SaveMetaArtifacts(symbol);
}

#endif // __FXAI_META_PERSISTENCE_MQH__
