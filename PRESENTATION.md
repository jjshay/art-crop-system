# Art Crop System - Presentation Guide

## Elevator Pitch (30 seconds)

> "The Art Crop System is a multi-tier AI pipeline that automatically detects and extracts artwork from photographs. When you photograph art on a wall, you get the frame, shadows, and background. This system uses a combination of background removal, multi-AI vision analysis, and computer vision to perfectly crop just the artwork—plus it generates 8 detail shots for e-commerce listings."

---

## Key Talking Points

### 1. The Problem It Solves

- **Manual Cropping is Time-Consuming**: Each artwork photo needs careful editing
- **Inconsistent Results**: Manual cropping varies in quality
- **Missing Detail Shots**: E-commerce needs corner crops, signature closeups
- **Scale Challenge**: Art dealers process hundreds of pieces monthly

### 2. The Solution

- **4-Tier Processing Pipeline**:
  - Tier 1: Fast background removal (rembg)
  - Tier 2: Multi-AI boundary consensus (GPT-4V, Claude, Gemini, Grok)
  - Tier 3: Computer vision fallback (OpenCV edge detection)
  - Tier 4: Quality validation

- **Automatic Detail Generation**: 8 standardized detail shots per artwork
- **Signature/Edition Detection**: Automatically locates artist signatures

### 3. Technical Architecture

```
Photo → rembg → Multi-AI Consensus → CV Fallback → Quality Check → Crops + Details
```

---

## Demo Script

### What to Show

1. **Run the Demo** (`python demo.py`)
   - Show a wall photo being processed
   - Walk through each tier of processing
   - Display the generated detail shots

2. **Key Moments to Pause**
   - When rembg produces a clean result (explain the fast path)
   - When AIs are consulted (show boundary consensus)
   - Final output with all 8 detail shots

3. **Sample Output Discussion**
   - Show `sample_output/crop_report.json`
   - Point out the boundary coordinates from each AI
   - Explain the quality score

---

## Technical Highlights to Mention

### Tiered Processing Strategy
- "Why waste expensive AI calls when simple background removal works 70% of the time?"
- "Each tier adds intelligence but also cost—we escalate only when needed"

### Multi-AI Consensus for Boundaries
- "4 AI vision models identify artwork boundaries independently"
- "Averaging coordinates reduces individual model errors"

### Detail Shot Generation
- "Automated corner crops catch condition issues"
- "Signature detection helps with artist attribution"

---

## Anticipated Questions & Answers

**Q: Why not just use one AI for cropping?**
> "Single models often hallucinate boundaries, especially with complex frames or reflective glass. Multi-model consensus is dramatically more accurate for edge cases."

**Q: What's the success rate?**
> "Tier 1 (rembg) handles about 70% of clean photos. With all tiers, we achieve >95% accuracy on correctly detecting artwork boundaries."

**Q: How do you handle framed vs. unframed art?**
> "The AI models are prompted to detect the actual artwork, not the frame. For framed pieces, we identify the mat/artwork edge, not the outer frame."

**Q: What about glass reflections?**
> "This is where multi-AI consensus really shines. Different models handle reflections differently, so averaging their results cancels out reflection-based errors."

---

## Key Metrics to Share

| Metric | Value |
|--------|-------|
| Processing Tiers | 4 (rembg → AI → CV → Quality) |
| AI Models | 4 (GPT-4V, Claude, Gemini, Grok) |
| Detail Shots Generated | 8 per artwork |
| Tier 1 Success Rate | ~70% |
| Overall Accuracy | >95% |

---

## Why This Project Matters

1. **Shows Practical AI Application**: Real business problem solved with AI
2. **Demonstrates Cost Optimization**: Tiered approach minimizes API costs
3. **Computer Vision Expertise**: Combines ML models with traditional CV
4. **Production Scale**: Designed for processing hundreds of images
5. **Domain Knowledge**: Understanding of art photography challenges

---

## Closing Statement

> "This project showcases my ability to design efficient AI pipelines that balance accuracy with cost. The tiered approach—using simple solutions first and escalating to expensive AI only when needed—is a pattern I apply across all my systems."
