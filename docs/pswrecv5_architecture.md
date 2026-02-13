# PSWRecV5 Architecture Diagram

## Mermaid Diagram (render in Markdown, GitHub, or [mermaid.live](https://mermaid.live))

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        IE[Item Embedding]
        PE[Position Embedding]
        LN1[LayerNorm]
        D1[Dropout]
    end

    subgraph PhaseFilter["LocalPhaseFilterBankV5"]
        direction TB
        B1[Band 1: k=3, d=1]
        B2[Band 2: k=7, d=2]
        B3[Band 3: k=15, d=4]
        B4[Band 4: k=31, d=8]
        B1 --> B2 --> B3 --> B4
        B4 --> CF[cos_phi, sin_phi]
        B4 --> MAG[mag]
    end

    subgraph Encoder["PSWEncoderV5 (N layers)"]
        direction TB
        subgraph Block["PSWBlockV5 × N"]
            subgraph PSA["PhaseSyncAttentionV5"]
                QKV[Q, K, V projections]
                SDP[Scaled dot-product scores]
                PB[+ Phase bias (harmonic-mean mag gated)]
                SM[Softmax]
                GATE["Post-softmax phase gating\nattn × sigmoid(scale × phase_scores)"]
                RENORM[Renormalize]
                CTX[Context = attn @ V]
                QKV --> SDP --> PB --> SM --> GATE --> RENORM --> CTX
            end
            subgraph FF["FeedForwardV5"]
                L1[Linear hidden→inner]
                ACT[GELU]
                L2[Linear inner→hidden]
                FF_RES[Residual + LayerNorm]
                L1 --> ACT --> L2 --> FF_RES
            end
            PSA --> FF
        end
    end

    subgraph Output["Output"]
        GATHER["Gather last position\n[seq_len-1]"]
        PRED[Full-sort: output @ item_embᵀ]
    end

    IE --> LN1
    PE --> LN1
    LN1 --> D1
    D1 --> PhaseFilter
    D1 --> Encoder
    PhaseFilter -->|cos_phi, sin_phi, mag| Encoder
    Encoder --> GATHER --> PRED
```

## Simplified Block Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PSWRecV5                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Item Seq ──► Item Emb + Pos Emb ──► LayerNorm + Dropout                     │
│       │                    │                                                 │
│       │                    ├──► LocalPhaseFilterBankV5 ──► cos_φ, sin_φ, mag │
│       │                    │                    │                            │
│       │                    ▼                    │                            │
│       │              ┌──────────────────────────┴────────────────────────┐   │
│       │              │  PSWBlockV5 × N                                   │   │
│       │              │  ┌─────────────────────────────────────────────┐  │   │
│       │              │  │ PhaseSyncAttentionV5                         │  │   │
│       │              │  │  Q,K,V ──► scores ──► + phase_bias ──► softmax│  │   │
│       │              │  │    ──► × gate(phase) ──► renormalize ──► @V   │  │   │
│       │              │  └─────────────────────────────────────────────┘  │   │
│       │              │  ┌─────────────────────────────────────────────┐  │   │
│       │              │  │ FeedForwardV5: Linear→GELU→Linear + resid   │  │   │
│       │              │  └─────────────────────────────────────────────┘  │   │
│       │              └──────────────────────────────────────────────────┘   │
│       │                                        │                            │
│       └────────────────────────────────────────┼──► [last] ──► logits/scores│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

| Stage | Input | Output |
|-------|-------|--------|
| Embedding | item_seq | input_emb [B, L, D] |
| Phase Filter | input_emb | cos_phi, sin_phi, mag [B, n_bands, L] |
| PhaseSyncAttention | hidden, cos_phi, sin_phi, mag | hidden [B, L, D] |
| FeedForward | hidden | hidden [B, L, D] |
| Output | encoder_output | output[last_pos] [B, D] |
