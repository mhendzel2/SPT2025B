# Quick Testing Guide - Navigation Fixes

## 🎯 What Was Fixed

Two critical bugs that crashed the app when trying to proceed through workflows:
- ✅ **Bug #8**: "Proceed to Tracking" button now works
- ✅ **Bug #9**: "Proceed to Image Processing" button now works
- ⚠️ **Bug #10**: Drag-and-drop investigation (needs your feedback)

---

## 🧪 Test 1: Tracking Workflow (Bug #8)

### Steps:
1. Start app: 
   ```powershell
   streamlit run app.py
   ```

2. Navigate: **Data Loading** tab → **Upload Images for Tracking** tab

3. Upload image:
   - Click "Browse files" or drag file
   - Use: `sample data/Image timelapse/Cell1.tif`

4. Click: **"Proceed to Tracking"** button

### ✅ Expected Result:
- App navigates to Tracking page
- Shows "Particle Detection and Tracking" title
- No crash or "Stopping..." message

### ❌ If It Fails:
- App crashes
- Console shows "Stopping..."
- Report back with error message

---

## 🧪 Test 2: Mask Generation Workflow (Bug #9)

### Steps:
1. In same app session, go to: **Data Loading** tab → **Upload Images for Mask Generation** tab

2. Upload image:
   - Click "Browse files" or drag file
   - Use: `sample data/Image Channels/Cell1.tif`

3. Click: **"Proceed to Image Processing"** button

### ✅ Expected Result:
- App navigates to Image Processing page
- Shows "Image Processing & Nuclear Density Analysis" title
- No crash

### ❌ If It Fails:
- App crashes
- Report back with error message

---

## 🧪 Test 3: Drag-and-Drop (Bug #10)

### Steps:
1. Go to: **Data Loading** → **Upload Images for Tracking**

2. **Test A**: Try dragging file from file explorer onto upload box
   - Does it highlight when you drag over it?
   - Does it accept the file?

3. **Test B**: Click "Browse files" button
   - Does file selection dialog open?
   - Does it accept the file?

### Please Report:
1. **Browser**: Chrome, Firefox, Edge, Safari, Other?
2. **Browser Version**: (Check in browser settings)
3. **Test A Result**: Drag-and-drop works? Yes/No
4. **Test B Result**: Browse button works? Yes/No
5. **File Tested**: Name and size
6. **Console Errors**: Press F12, check Console tab, copy any red errors

---

## 📊 Results Template

Copy and fill this out:

```
TEST 1 - TRACKING NAVIGATION:
Status: [ ] Pass [ ] Fail
Notes: 

TEST 2 - MASK NAVIGATION:
Status: [ ] Pass [ ] Fail
Notes: 

TEST 3 - FILE UPLOAD:
Browser: 
Version: 
Drag-and-drop: [ ] Works [ ] Doesn't work
Browse button: [ ] Works [ ] Doesn't work
Console errors: 

```

---

## 🔧 Quick Troubleshooting

### If Navigation Still Fails:
1. Fully restart the app (Ctrl+C, then restart)
2. Clear browser cache (Ctrl+Shift+Del)
3. Try incognito/private window
4. Check console for errors (F12)

### If Drag-and-Drop Doesn't Work:
1. ✅ Use "Browse files" button instead (should work)
2. Try different browser (Chrome recommended)
3. Check file extension: Must be .tif, .tiff, .png, .jpg, or .jpeg
4. Check file size: Must be under 200 MB

### If Everything Still Fails:
1. Run test script:
   ```powershell
   python test_session_state_fixes.py
   ```
2. Should show: "🎉 ALL TESTS PASSED!"
3. If not, report the output

---

## ✨ What Should Work Now

### Before Fixes:
```
User clicks "Proceed to Tracking" → App crashes with "Stopping..."
User clicks "Proceed to Image Processing" → App crashes
```

### After Fixes:
```
User clicks "Proceed to Tracking" → ✅ Navigates to Tracking page
User clicks "Proceed to Image Processing" → ✅ Navigates to Image Processing page
User can complete full workflow without crashes
```

---

## 📞 What to Report Back

### ✅ If Tests Pass:
Just say: "All navigation tests passed!"

### ⚠️ If Tests Fail:
1. Which test failed? (#1, #2, or #3)
2. Exact error message from console
3. Screenshot helpful but not required
4. Copy the console output (if any)

### ℹ️ For Drag-and-Drop:
Fill out the "Test 3" section above with browser details

---

## 🎓 Understanding the Fixes

### What Caused the Crashes?

**Before**:
```python
if st.session_state.image_data is None:  # ❌ KeyError if key doesn't exist
```

**After**:
```python
if 'image_data' not in st.session_state or st.session_state.image_data is None:  # ✅ Safe
```

The code was trying to access a dictionary key without checking if it existed first!

### Why It Matters

These are **critical workflow paths** - users can't proceed without these buttons working. The fixes ensure:
1. Defensive programming (check before access)
2. Graceful error handling (show helpful message)
3. Smooth user experience (no unexpected crashes)

---

**Ready to test? Start with Test 1!** 🚀
