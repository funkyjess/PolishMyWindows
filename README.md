# PolishMyWindows
 
A safe, lightweight file organizer for Windows.
 
## Requirements
 
- Python 3.9+ (Windows x64)
 
## Usage
 
Dry-run (prints what it would do, does not move files):
 
```powershell
python .\polishmywindows.py organize "C:\Path\To\Folder" --print-plan
```
 
Actually move files:
 
```powershell
python .\polishmywindows.py organize "C:\Path\To\Folder" --print-plan --apply
```
 
Undo recent moves:
 
```powershell
python .\polishmywindows.py undo "C:\Path\To\Folder" --dry-run
python .\polishmywindows.py undo "C:\Path\To\Folder" --steps 50
```
 
## Rules
 
Categories and extensions live in `rules.json`.
