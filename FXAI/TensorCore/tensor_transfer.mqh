#ifndef __FXAI_TENSOR_TRANSFER_MQH__
#define __FXAI_TENSOR_TRANSFER_MQH__

int FXAI_SharedTransferDomainBucket(const double domain_hash)
{
   double v = FXAI_Clamp(domain_hash, 0.0, 1.0 - 1e-9);
   int bucket = (int)MathFloor(v * (double)FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS);
   if(bucket < 0) bucket = 0;
   if(bucket >= FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS) bucket = FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS - 1;
   return bucket;
}

int FXAI_SharedTransferHorizonBucket(const int horizon_minutes)
{
   int h = horizon_minutes;
   if(h < 1) h = 1;
   if(h > 1440) h = 1440;
   int slot = 0;
   if(h <= 2) slot = 0;
   else if(h <= 5) slot = 1;
   else if(h <= 15) slot = 2;
   else if(h <= 30) slot = 3;
   else if(h <= 60) slot = 4;
   else if(h <= 240) slot = 5;
   else if(h <= 720) slot = 6;
   else slot = 7;
   if(slot < 0) slot = 0;
   if(slot >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS)
      slot = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;
   return slot;
}

void FXAI_SharedTransferEncode(const double &a[],
                               const int domain_bucket,
                               const int horizon_bucket,
                               const int session_bucket,
                               const double &w[][FXAI_SHARED_TRANSFER_FEATURES],
                               const double &b[],
                               const double &domain_emb[][FXAI_SHARED_TRANSFER_LATENT],
                               const double &horizon_emb[][FXAI_SHARED_TRANSFER_LATENT],
                               const double &session_emb[][FXAI_SHARED_TRANSFER_LATENT],
                               double &latent[])
{
   ArrayResize(latent, FXAI_SHARED_TRANSFER_LATENT);
   int db = domain_bucket;
   if(db < 0) db = 0;
   if(db >= FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS) db = FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS - 1;
   int hb = horizon_bucket;
   if(hb < 0) hb = 0;
   if(hb >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS) hb = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;
   int sb = session_bucket;
   if(sb < 0) sb = 0;
   if(sb >= FXAI_PLUGIN_SESSION_BUCKETS) sb = FXAI_PLUGIN_SESSION_BUCKETS - 1;

   for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
   {
      double z = b[j] +
                 domain_emb[db][j] +
                 horizon_emb[hb][j] +
                 session_emb[sb][j];
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         z += w[j][i] * a[i];
      latent[j] = FXAI_Tanh(FXAI_ClipSym(z, 6.0));
   }
}

void FXAI_SharedTransferSoftmax(const double &logits[],
                                double &probs[])
{
   ArrayResize(probs, 3);
   double mx = logits[0];
   if(logits[1] > mx) mx = logits[1];
   if(logits[2] > mx) mx = logits[2];

   double den = 0.0;
   for(int c=0; c<3; c++)
   {
      probs[c] = MathExp(FXAI_ClipSym(logits[c] - mx, 10.0));
      den += probs[c];
   }
   if(den <= 0.0) den = 1.0;
   for(int c=0; c<3; c++)
      probs[c] /= den;
}

#endif // __FXAI_TENSOR_TRANSFER_MQH__
