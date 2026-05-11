# FPS Monitoring Guide for Cat Behavior Data Collection

## Overview
This guide explains the FPS monitoring features added to `collect_cat_behavior.py` to ensure high-quality data collection.

## Key Concepts

### Why FPS Matters
- **Inference FPS** = How fast the YOLO model processes frames
- **Video FPS** = Original video frame rate (typically 25-30 fps)
- For best results: **Inference FPS should be 1.2-1.5× Video FPS**

### Why Higher Inference FPS?
1. **No Frame Drops**: Ensures every video frame is processed
2. **Consistent Timing**: 30-frame sequences maintain accurate 1-second duration
3. **Stable Features**: No missing frames in the sequence buffer
4. **Better Quality**: Complete movement patterns captured

## FPS Color Indicators

| Color | Status | Meaning | Action Needed |
|-------|--------|---------|---------------|
| 🟢 Green | GOOD | FPS ≥ Target (1.3× video FPS) | ✅ Perfect! Continue collecting |
| 🟡 Yellow | OK | FPS ≥ Video FPS | ⚠️ Acceptable, but watch for drops |
| 🔴 Red | LOW | FPS < Video FPS | ❌ Will drop frames - optimize! |

## On-Screen Display

### During Collection:
```
Top-right corner:
┌─────────────────┐
│ FPS:28.5 GOOD  │ ← Color-coded
│ Target:32+      │ ← Aim for this
└─────────────────┘
```

### At Program End:
```
📊 FPS Statistics:
  Average: 28.5 fps
  Stability: ±2.1 fps
  Video FPS: 25.0 fps
  Target FPS: 32.5 fps
  ✅ Performance: EXCELLENT
```

## Performance Optimization

### If FPS is RED or YELLOW:

1. **Reduce IMGSZ** (most effective):
   ```python
   IMGSZ = 480  # Default: 640
   # Or even lower:
   IMGSZ = 320
   ```

2. **Close other programs**:
   - Web browsers with many tabs
   - Video players
   - Resource-heavy applications

3. **Check system resources**:
   - CPU usage should be < 80%
   - GPU usage (if available) should have headroom

4. **Simplify visualization** (if needed):
   - Comment out skeleton drawing temporarily
   - Reduce window size

## Data Quality Impact

### Excellent FPS (Green):
- All frames processed
- Sequences are exactly 30 frames @ 1 second
- Movement patterns complete and smooth
- Best training results

### Acceptable FPS (Yellow):
- Occasional frame drops possible
- Most sequences still valid
- Minor gaps in movement
- Still usable for training

### Low FPS (Red):
- Frequent frame drops
- Sequences have timing issues
- Missing critical movement frames
- Poor training results - **DO NOT USE**

## Recommended Settings by System

### High-end PC (GPU):
```python
IMGSZ = 640
# Expected: 40-60 FPS
```

### Mid-range PC:
```python
IMGSZ = 480
# Expected: 25-35 FPS
```

### Low-end PC:
```python
IMGSZ = 320
# Expected: 15-25 FPS
# Consider using lower FPS video source
```

## Monitoring Tips

1. **Let it stabilize**: FPS may fluctuate in first 5-10 seconds
2. **Watch the average**: Short dips are OK, sustained low FPS is not
3. **Stability matters**: ±5 fps variation is good, ±15 fps is concerning
4. **Check end statistics**: Tells you if collection session was good quality

## Example Sessions

### Good Session:
```
🎬 Video FPS: 25.0
🎯 Target FPS: 32.5+ (for continuous capture)

📊 FPS Statistics:
  Average: 34.2 fps
  Stability: ±2.8 fps
  ✅ Performance: EXCELLENT
```
→ **Use this data for training!**

### Marginal Session:
```
🎬 Video FPS: 30.0
🎯 Target FPS: 39.0+ (for continuous capture)

📊 FPS Statistics:
  Average: 31.5 fps
  Stability: ±4.2 fps
  ⚠️ Performance: ACCEPTABLE (may drop frames)
```
→ **Usable, but optimize for next session**

### Bad Session:
```
🎬 Video FPS: 25.0
🎯 Target FPS: 32.5+ (for continuous capture)

📊 FPS Statistics:
  Average: 18.3 fps
  Stability: ±6.7 fps
  ❌ Performance: LOW (will drop frames)
  💡 Tip: Reduce IMGSZ or simplify visualization
```
→ **DO NOT USE - re-collect with optimized settings**

## Troubleshooting

### FPS starts good but drops over time
- **Cause**: Thermal throttling or memory leak
- **Solution**: Take breaks, close/restart program

### FPS fluctuates wildly
- **Cause**: Background processes competing for resources
- **Solution**: Close unnecessary programs, check Task Manager

### FPS always red even with IMGSZ=320
- **Cause**: Very slow system or complex video
- **Solution**: Use simpler video source (lower resolution/FPS)

## Quick Checklist Before Collection

- [ ] IMGSZ set appropriately for your system
- [ ] Other programs closed
- [ ] Run test session first to check FPS
- [ ] Wait 10 seconds for FPS to stabilize
- [ ] Check FPS indicator is GREEN or YELLOW
- [ ] Note end statistics for quality assurance

## Summary

**Target**: Inference FPS ≥ 1.3× Video FPS (Green indicator)
**Minimum**: Inference FPS ≥ Video FPS (Yellow indicator)  
**Invalid**: Inference FPS < Video FPS (Red indicator)

Keep FPS in GREEN zone for best results! 🚀
