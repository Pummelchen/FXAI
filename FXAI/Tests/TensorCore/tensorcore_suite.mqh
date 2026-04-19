#ifndef __FXAI_TENSORCORE_SUITE_MQH__
#define __FXAI_TENSORCORE_SUITE_MQH__

#include "..\TestHarness\test_harness.mqh"
#include "..\..\TensorCore\TensorCore.mqh"

FXAISequenceBuffer g_fxai_tensor_suite_buffer;
FXAISequenceBuffer g_fxai_tensor_suite_buffer_copy;
FXAISequenceRuntimeState g_fxai_tensor_suite_rt_state;
FXAISequenceRuntimeState g_fxai_tensor_suite_rt_copy;
FXAISequenceRuntimeState g_fxai_tensor_suite_push_state;
double g_fxai_tensor_suite_current_x[FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_w_in[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_mm_out[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_attn_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_conv_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_ffn_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_seq_future[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_conv_future[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_attn_future[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_mix[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
double g_fxai_tensor_suite_mat_p[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_mat_m[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_mat_v[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_mat_g[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_w_plus[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_w_minus[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_mm_plus[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_MLP_HIDDEN];
double g_fxai_tensor_suite_mm_minus[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_MLP_HIDDEN];

bool FXAI_TensorCoreTestKernelBlocks(string &reason)
{
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      g_fxai_tensor_suite_current_x[k] = (k == 0 ? 1.0 : 0.02 * (double)(k + 1));

   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         g_fxai_tensor_suite_window[t][k] = (k == 0 ? 1.0 : 0.01 * (double)(t + 1) * (double)(k + 1));
   }

   FXAI_SequenceBufferLoadWindow(g_fxai_tensor_suite_buffer,
                                 g_fxai_tensor_suite_current_x,
                                 g_fxai_tensor_suite_window,
                                 5,
                                 6,
                                 false);
   if(g_fxai_tensor_suite_buffer.len != 6 ||
      g_fxai_tensor_suite_buffer.mask[0] != 1 ||
      g_fxai_tensor_suite_buffer.mask[5] != 1)
   {
      reason = "sequence_buffer_load";
      return false;
   }
   if(!(g_fxai_tensor_suite_buffer.pos_bias[0] < g_fxai_tensor_suite_buffer.pos_bias[5] &&
        MathAbs(g_fxai_tensor_suite_buffer.pos_bias[5]) < 1e-9))
   {
      reason = "sequence_positional_bias";
      return false;
   }

   FXAI_SequenceBufferCopy(g_fxai_tensor_suite_buffer, g_fxai_tensor_suite_buffer_copy);
   if(g_fxai_tensor_suite_buffer_copy.len != g_fxai_tensor_suite_buffer.len ||
      MathAbs(g_fxai_tensor_suite_buffer_copy.data[2][3] - g_fxai_tensor_suite_buffer.data[2][3]) > 1e-12)
   {
      reason = "sequence_copy";
      return false;
   }

   int seq_len = 0;
   int seq_mask[];
   double seq_pos[];
   FXAI_SequenceBufferPreparePacked(g_fxai_tensor_suite_buffer, g_fxai_tensor_suite_seq, seq_len, seq_mask, seq_pos);
   if(seq_len != 6 || ArraySize(seq_mask) != FXAI_MAX_SEQUENCE_BARS || ArraySize(seq_pos) != FXAI_MAX_SEQUENCE_BARS)
   {
      reason = "sequence_pack";
      return false;
   }

   FXAISequenceRuntimeConfig rt_cfg = FXAI_SequenceRuntimeMakeConfig(4, 2, 2, true, true, 0.04);
   FXAI_SequenceRuntimeReset(g_fxai_tensor_suite_rt_state, rt_cfg);
   FXAI_SequenceRuntimeLoadWindow(g_fxai_tensor_suite_rt_state,
                                  g_fxai_tensor_suite_current_x,
                                  g_fxai_tensor_suite_window,
                                  5);
   if(g_fxai_tensor_suite_rt_state.buffer.len < 2 || g_fxai_tensor_suite_rt_state.buffer.len > 4)
   {
      reason = "sequence_runtime_load";
      return false;
   }
   FXAI_SequenceRuntimeCopy(g_fxai_tensor_suite_rt_state, g_fxai_tensor_suite_rt_copy);
   if(g_fxai_tensor_suite_rt_copy.steps_seen != g_fxai_tensor_suite_rt_state.steps_seen ||
      g_fxai_tensor_suite_rt_copy.buffer.len != g_fxai_tensor_suite_rt_state.buffer.len ||
      g_fxai_tensor_suite_rt_copy.raw_len != g_fxai_tensor_suite_rt_state.raw_len)
   {
      reason = "sequence_runtime_copy";
      return false;
   }

   FXAISequenceRuntimeConfig push_cfg = FXAI_SequenceRuntimeMakeConfig(4, 2, 2, false, true, 0.04);
   FXAI_SequenceRuntimeReset(g_fxai_tensor_suite_push_state, push_cfg);
   for(int step=0; step<5; step++)
   {
      double push_x[FXAI_AI_WEIGHTS];
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         push_x[i] = (i == 0 ? 1.0 : 0.05 * (double)(step + 1) * (double)(i + 1));
      FXAI_SequenceRuntimePush(g_fxai_tensor_suite_push_state, push_x);
   }
   if(g_fxai_tensor_suite_push_state.raw_len != 5 ||
      g_fxai_tensor_suite_push_state.buffer.len != 2 ||
      g_fxai_tensor_suite_push_state.buffer.mask[0] != 1 ||
      g_fxai_tensor_suite_push_state.buffer.mask[1] != 1)
   {
      reason = "sequence_runtime_push";
      return false;
   }

   double b_in[FXAI_AI_MLP_HIDDEN];
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         g_fxai_tensor_suite_w_in[i][o] = ((i < 6 && o < 4) ? 0.01 * (double)(i + 1) * (double)(o + 1) : 0.0);
   for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      b_in[o] = (o < 4 ? 0.05 * (double)(o + 1) : 0.0);

   FXAI_TensorBatchedMatMul(g_fxai_tensor_suite_seq,
                            seq_len,
                            g_fxai_tensor_suite_w_in,
                            b_in,
                            g_fxai_tensor_suite_mm_out,
                            4);
   double expect0 = b_in[0];
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      expect0 += g_fxai_tensor_suite_seq[0][i] * g_fxai_tensor_suite_w_in[i][0];
   if(MathAbs(g_fxai_tensor_suite_mm_out[0][0] - expect0) > 1e-8)
   {
      reason = "batched_matmul_forward";
      return false;
   }

   double gemv_out[];
   FXAI_ModuleLinearInputHidden(g_fxai_tensor_suite_current_x,
                                g_fxai_tensor_suite_w_in,
                                b_in,
                                gemv_out,
                                4);
   if(ArraySize(gemv_out) != 4 || MathAbs(gemv_out[0] - expect0) > 1e-8)
   {
      reason = "linear_module_forward";
      return false;
   }

   double hh_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double hh_b[FXAI_AI_MLP_HIDDEN];
   double hh_in[FXAI_AI_MLP_HIDDEN];
   for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
   {
      hh_in[i] = 0.03 * (double)(i + 1);
      hh_b[i] = (i < 3 ? 0.02 * (double)(i + 1) : 0.0);
      for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         hh_w[i][j] = ((i < 4 && j < 3) ? 0.015 * (double)(i + 1) * (double)(j + 1) : 0.0);
   }
   double hh_out[];
   FXAI_ModuleLinearHidden(hh_in, hh_w, hh_b, hh_out, 3);
   if(ArraySize(hh_out) != 3 || !MathIsValidNumber(hh_out[2]))
   {
      reason = "linear_hidden_forward";
      return false;
   }

   double kernel[3] = {0.25, 0.50, 0.25};
   double conv[];
   FXAI_ModuleConv1DSummary(g_fxai_tensor_suite_seq, seq_len, kernel, 3, conv);
   double conv_ref = 0.0;
   double conv_den = 0.0;
   for(int t=0; t<seq_len; t++)
   {
      double y = 0.0;
      double kw = 0.0;
      for(int j=0; j<3; j++)
      {
         int idx = t - j;
         if(idx < 0)
            break;
         y += kernel[j] * g_fxai_tensor_suite_seq[idx][1];
         kw += MathAbs(kernel[j]);
      }
      if(kw <= 0.0)
         kw = 1.0;
      conv_ref += y / kw;
      conv_den += 1.0;
   }
   conv_ref /= MathMax(conv_den, 1.0);
   if(MathAbs(conv[1] - conv_ref) > 1e-8)
   {
      reason = "conv1d_forward";
      return false;
   }

   double query[FXAI_AI_WEIGHTS];
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      query[i] = g_fxai_tensor_suite_seq[seq_len - 1][i];
   double attn[];
   FXAI_ModuleMultiHeadAttentionSummary(g_fxai_tensor_suite_seq, seq_len, query, seq_mask, seq_pos, 2, attn);
   if(ArraySize(attn) != FXAI_AI_WEIGHTS || !MathIsValidNumber(attn[2]))
   {
      reason = "attention_module_forward";
      return false;
   }

   FXAITensorDims dims = FXAI_TensorMakeDims(12, FXAI_AI_MLP_HIDDEN, 2, 6, 1, 1, 1, 0.05);
   double pooled[];
   FXAI_ModuleSequenceAttentionBlock(g_fxai_tensor_suite_seq, seq_len, seq_mask, seq_pos, dims, g_fxai_tensor_suite_attn_seq);
   FXAI_ModuleSequenceConvBlock(g_fxai_tensor_suite_seq, seq_len, kernel, 3, dims, g_fxai_tensor_suite_conv_seq);
   FXAI_ModuleSequenceResidualNormFFN(g_fxai_tensor_suite_attn_seq, seq_len, dims, g_fxai_tensor_suite_ffn_seq);
   FXAI_ModuleSequencePool(g_fxai_tensor_suite_ffn_seq, seq_len, dims, pooled);
   if(ArraySize(pooled) != FXAI_AI_WEIGHTS ||
      !MathIsValidNumber(g_fxai_tensor_suite_attn_seq[seq_len - 1][2]) ||
      !MathIsValidNumber(g_fxai_tensor_suite_conv_seq[seq_len - 1][2]) ||
      !MathIsValidNumber(g_fxai_tensor_suite_ffn_seq[seq_len - 1][2]))
   {
      reason = "sequence_blocks_forward";
      return false;
   }

   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         g_fxai_tensor_suite_seq_future[t][i] = g_fxai_tensor_suite_seq[t][i];
   g_fxai_tensor_suite_seq_future[seq_len - 1][1] += 10.0;
   FXAI_ModuleSequenceConvBlock(g_fxai_tensor_suite_seq_future, seq_len, kernel, 3, dims, g_fxai_tensor_suite_conv_future);
   FXAI_ModuleSequenceAttentionBlock(g_fxai_tensor_suite_seq_future, seq_len, seq_mask, seq_pos, dims, g_fxai_tensor_suite_attn_future);
   if(MathAbs(g_fxai_tensor_suite_conv_future[0][1] - g_fxai_tensor_suite_conv_seq[0][1]) > 1e-9)
   {
      reason = "sequence_conv_causality";
      return false;
   }
   if(MathAbs(g_fxai_tensor_suite_attn_future[0][1] - g_fxai_tensor_suite_attn_seq[0][1]) > 1e-9)
   {
      reason = "sequence_attention_causality";
      return false;
   }

   double ln_vec[4] = {1.0, 2.0, 3.0, 4.0};
   double ones[4] = {1.0, 1.0, 1.0, 1.0};
   double zeros[4] = {0.0, 0.0, 0.0, 0.0};
   FXAI_ModuleLayerNormAffine(ln_vec, 4, ones, zeros);
   double ln_mean = 0.0;
   double ln_var = 0.0;
   for(int i=0; i<4; i++)
      ln_mean += ln_vec[i];
   ln_mean /= 4.0;
   for(int i=0; i<4; i++)
   {
      double d = ln_vec[i] - ln_mean;
      ln_var += d * d;
   }
   ln_var /= 4.0;
   if(MathAbs(ln_mean) > 1e-6 || MathAbs(ln_var - 1.0) > 1e-3)
   {
      reason = "layernorm_forward";
      return false;
   }

   double rms_vec[4] = {1.0, 2.0, 3.0, 4.0};
   FXAI_ModuleRMSNormAffine(rms_vec, 4, ones, zeros);
   double rms_sq = 0.0;
   for(int i=0; i<4; i++)
      rms_sq += rms_vec[i] * rms_vec[i];
   rms_sq /= 4.0;
   if(MathAbs(rms_sq - 1.0) > 1e-3)
   {
      reason = "rmsnorm_forward";
      return false;
   }

   double i_gate[FXAI_AI_MLP_HIDDEN], f_gate[FXAI_AI_MLP_HIDDEN], o_gate[FXAI_AI_MLP_HIDDEN], g_gate[FXAI_AI_MLP_HIDDEN];
   double c_prev[FXAI_AI_MLP_HIDDEN], c_next[FXAI_AI_MLP_HIDDEN], h_next[FXAI_AI_MLP_HIDDEN];
   double z_gate[FXAI_AI_MLP_HIDDEN], r_gate[FXAI_AI_MLP_HIDDEN], n_gate[FXAI_AI_MLP_HIDDEN], h_prev[FXAI_AI_MLP_HIDDEN], gru_next[FXAI_AI_MLP_HIDDEN];
   for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
   {
      i_gate[h] = 0.4;
      f_gate[h] = 0.5;
      o_gate[h] = 0.6;
      g_gate[h] = 0.2;
      c_prev[h] = 0.1;
      z_gate[h] = 0.3;
      r_gate[h] = 0.4;
      n_gate[h] = 0.2;
      h_prev[h] = 0.1;
   }
   FXAI_ModuleLSTMCellForward(i_gate, f_gate, o_gate, g_gate, c_prev, c_next, h_next, 3);
   if(MathAbs(c_next[0] - (0.5 * 0.1 + 0.4 * 0.2)) > 1e-8)
   {
      reason = "lstm_cell_forward";
      return false;
   }
   FXAI_ModuleGRUCellForward(z_gate, r_gate, n_gate, h_prev, gru_next, 3);
   if(!MathIsValidNumber(gru_next[0]))
   {
      reason = "gru_cell_forward";
      return false;
   }

   double decay[FXAI_AI_MLP_HIDDEN];
   double skip[FXAI_AI_MLP_HIDDEN];
   double state_prev[FXAI_AI_MLP_HIDDEN], state_next[FXAI_AI_MLP_HIDDEN], ss_out[];
   for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
   {
      decay[h] = 0.8;
      skip[h] = 0.05;
      state_prev[h] = 0.1;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         g_fxai_tensor_suite_mix[h][i] = (h < 2 && i < 6 ? 0.02 * (double)(h + 1) * (double)(i + 1) : 0.0);
   }
   FXAI_ModuleStateSpaceBlockForward(g_fxai_tensor_suite_current_x,
                                     state_prev,
                                     decay,
                                     g_fxai_tensor_suite_mix,
                                     skip,
                                     state_next,
                                     ss_out,
                                     3);
   if(ArraySize(ss_out) != 3 || !MathIsValidNumber(ss_out[1]))
   {
      reason = "state_space_forward";
      return false;
   }

   double mix_a[3] = {1.0, 2.0, 3.0};
   double mix_b[3] = {3.0, 2.0, 1.0};
   double mix_c[3] = {2.0, 2.0, 2.0};
   double weights[3] = {0.2, 0.3, 0.5};
   double mix_out[];
   FXAI_ModuleGateBlend(mix_a, mix_b, 0.25, mix_out, 3);
   if(MathAbs(mix_out[0] - 1.5) > 1e-8)
   {
      reason = "gate_blend_forward";
      return false;
   }
   FXAI_ModuleMixtureFuse3(mix_a, mix_b, mix_c, weights, mix_out, 3);
   if(MathAbs(mix_out[0] - (0.2 * 1.0 + 0.3 * 3.0 + 0.5 * 2.0)) > 1e-8)
   {
      reason = "mixture_forward";
      return false;
   }

   double mse = FXAI_LossMSE(1.5, 1.0);
   double mse_grad = FXAI_LossMSEGrad(1.5, 1.0);
   if(MathAbs(mse - 0.25) > 1e-12 || MathAbs(mse_grad - 1.0) > 1e-12)
   {
      reason = "mse_loss";
      return false;
   }

   double ce_probs[3] = {0.2, 0.7, 0.1};
   double ce_grad[];
   FXAI_LossCrossEntropyGrad3(ce_probs, 1, ce_grad);
   if(ArraySize(ce_grad) != 3 || MathAbs(ce_grad[1] + 0.3) > 1e-8)
   {
      reason = "cross_entropy_grad";
      return false;
   }

   double vec_p[4] = {1.0, -0.5, 0.25, -0.125};
   double vec_v[4] = {0.0, 0.0, 0.0, 0.0};
   double vec_g[4] = {0.5, -0.25, 0.75, -1.0};
   FXAIParamGroupConfig group_cfg = FXAI_ParamGroupMakeConfig(0.01, 0.01, 1.5, 0.9, 0.999, 0.95, 0.5, 3);
   FXAIParamGroupStats group_stats;
   FXAI_OptSGDVectorStep(vec_p, vec_v, vec_g, 4, group_cfg, group_stats);
   if(group_stats.count != 4 || !(group_stats.grad_norm > 0.0) || !(group_stats.clip_scale > 0.0) || !MathIsValidNumber(vec_p[0]))
   {
      reason = "param_group_vector";
      return false;
   }

   for(int r=0; r<FXAI_AI_WEIGHTS; r++)
   {
      for(int c=0; c<FXAI_AI_MLP_HIDDEN; c++)
      {
         g_fxai_tensor_suite_mat_p[r][c] = ((r < 3 && c < 3) ? 0.05 * (double)(r + c + 1) : 0.0);
         g_fxai_tensor_suite_mat_m[r][c] = 0.0;
         g_fxai_tensor_suite_mat_v[r][c] = 0.0;
         g_fxai_tensor_suite_mat_g[r][c] = ((r < 3 && c < 3) ? 0.01 * (double)(r + 1) * (double)(c + 1) : 0.0);
      }
   }
   FXAI_OptAdamWMatrixInputHiddenStep(g_fxai_tensor_suite_mat_p,
                                      g_fxai_tensor_suite_mat_m,
                                      g_fxai_tensor_suite_mat_v,
                                      g_fxai_tensor_suite_mat_g,
                                      3,
                                      3,
                                      group_cfg,
                                      group_stats);
   if(group_stats.count != 9 ||
      !MathIsValidNumber(g_fxai_tensor_suite_mat_p[2][2]) ||
      !MathIsValidNumber(g_fxai_tensor_suite_mat_m[2][2]) ||
      !MathIsValidNumber(g_fxai_tensor_suite_mat_v[2][2]))
   {
      reason = "param_group_matrix";
      return false;
   }

   double clip_vec[4] = {3.0, 4.0, 0.0, 0.0};
   FXAI_ClipVectorInPlace(clip_vec, 4, 4.0);
   double clip_norm = FXAI_VectorNorm(clip_vec, 4);
   if(MathAbs(clip_norm - 4.0) > 1e-6)
   {
      reason = "gradient_clip";
      return false;
   }

   double lr_step = FXAI_LRScheduleStepDecay(0.1, 20, 10, 0.5, 0.001);
   double lr_cos = FXAI_LRScheduleCosineWarm(0.1, 100, 16, 128, 0.2);
   double lr_inv = FXAI_LRScheduleInvSqrt(0.1, 100, 16, 0.001);
   if(!(lr_step > 0.0 && lr_cos > 0.0 && lr_inv > 0.0))
   {
      reason = "lr_schedule";
      return false;
   }

   double eps = 1e-5;
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
   {
      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         g_fxai_tensor_suite_w_plus[i][o] = g_fxai_tensor_suite_w_in[i][o];
         g_fxai_tensor_suite_w_minus[i][o] = g_fxai_tensor_suite_w_in[i][o];
      }
   }
   g_fxai_tensor_suite_w_plus[1][0] += eps;
   g_fxai_tensor_suite_w_minus[1][0] -= eps;
   FXAI_TensorBatchedMatMul(g_fxai_tensor_suite_seq,
                            seq_len,
                            g_fxai_tensor_suite_w_plus,
                            b_in,
                            g_fxai_tensor_suite_mm_plus,
                            2);
   FXAI_TensorBatchedMatMul(g_fxai_tensor_suite_seq,
                            seq_len,
                            g_fxai_tensor_suite_w_minus,
                            b_in,
                            g_fxai_tensor_suite_mm_minus,
                            2);
   double loss_plus = 0.0;
   double loss_minus = 0.0;
   for(int t=0; t<seq_len; t++)
   {
      loss_plus += g_fxai_tensor_suite_mm_plus[t][0];
      loss_minus += g_fxai_tensor_suite_mm_minus[t][0];
   }
   double grad_fd = (loss_plus - loss_minus) / (2.0 * eps);
   double grad_ref = 0.0;
   for(int t=0; t<seq_len; t++)
      grad_ref += g_fxai_tensor_suite_seq[t][1];
   if(MathAbs(grad_fd - grad_ref) > 1e-4)
   {
      reason = "batched_matmul_gradient";
      return false;
   }

   string file_name = "FXAI\\runtime_sanity.bin";
   FolderCreate("FXAI", FILE_COMMON);
   int hw = FileOpen(file_name, FILE_BIN | FILE_WRITE | FILE_COMMON);
   if(hw == INVALID_HANDLE)
   {
      reason = "serialize_open_write";
      return false;
   }
   double params[4] = {vec_p[0], lr_cos, mix_out[0], grad_fd};
   for(int i=0; i<4; i++)
      FileWriteDouble(hw, params[i]);
   FileClose(hw);

   int hr = FileOpen(file_name, FILE_BIN | FILE_READ | FILE_COMMON);
   if(hr == INVALID_HANDLE)
   {
      reason = "serialize_open_read";
      return false;
   }
   for(int i=0; i<4; i++)
   {
      double value = FileReadDouble(hr);
      if(MathAbs(value - params[i]) > 1e-12)
      {
         FileClose(hr);
         FileDelete(file_name, FILE_COMMON);
         reason = "serialize_drift";
         return false;
      }
   }
   FileClose(hr);
   FileDelete(file_name, FILE_COMMON);

   reason = "";
   return true;
}

bool FXAI_TensorCoreTestAdamWConvergence(string &reason)
{
   double param = 3.0;
   double mean = 0.0;
   double var = 0.0;
   double start_abs = MathAbs(param);
   for(int step=1; step<=96; step++)
   {
      double grad = 2.0 * param;
      FXAI_OptAdamWStep(param, mean, var, grad, 0.05, 0.9, 0.999, 0.0005, step);
      if(!MathIsValidNumber(param) || !MathIsValidNumber(mean) || !MathIsValidNumber(var))
      {
         reason = "adamw_invalid_number";
         return false;
      }
   }
   if(!(MathAbs(param) < 0.10 * start_abs))
   {
      reason = "adamw_no_convergence";
      return false;
   }
   reason = "";
   return true;
}

bool FXAI_TensorCoreTestRMSPropConvergence(string &reason)
{
   double param = -2.5;
   double cache = 0.0;
   double start_abs = MathAbs(param);
   for(int step=0; step<128; step++)
   {
      double grad = 2.0 * param;
      FXAI_OptRMSPropStep(param, cache, grad, 0.02, 0.95, 0.0005);
      if(!MathIsValidNumber(param) || !MathIsValidNumber(cache))
      {
         reason = "rmsprop_invalid_number";
         return false;
      }
   }
   if(!(MathAbs(param) < 0.12 * start_abs))
   {
      reason = "rmsprop_no_convergence";
      return false;
   }
   reason = "";
   return true;
}

void FXAI_TensorCoreRunSuite(FXAITestSuiteResult &suite)
{
   FXAI_TestSuiteReset(suite, "tensorcore");

   string reason = "";
   bool passed = FXAI_TensorCoreTestKernelBlocks(reason);
   FXAI_TestSuiteAddCase(suite, "tensor_ops_sequence_and_normalization", passed, reason);

   reason = "";
   passed = FXAI_TensorCoreTestAdamWConvergence(reason);
   FXAI_TestSuiteAddCase(suite, "optimizer_adamw_convergence", passed, reason);

   reason = "";
   passed = FXAI_TensorCoreTestRMSPropConvergence(reason);
   FXAI_TestSuiteAddCase(suite, "optimizer_rmsprop_convergence", passed, reason);
}

#endif // __FXAI_TENSORCORE_SUITE_MQH__
