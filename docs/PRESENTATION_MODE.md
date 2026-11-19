# Presentation mode — Project Use

Selected by the user after reviewing sitting and standing demonstrations on
19 November 2026. The single retained desktop shortcut remains named
GestureDeck - Learned Trial, but now launches only palm/back navigation through
GestureDeck-Learned Trial.pyw using .venv/Scripts/pythonw.exe. No new launcher,
shortcut or terminal window is created.

The presentation interface no longer offers Sideways swipe or Learned swipe
trial. It retains camera preview, recognition/dry run and PowerPoint/live keys.
The synthetic swipe demo is omitted because it does not simulate palm/back turns.
The initial mode is dry run, session stopped, recording off, real keys unchecked.
The navigation option and window title say Project Use. This is the renamed
right-hand palm/back-turn mode; its gesture behavior is unchanged.

For the PowerPoint demonstration:

1. Open the existing desktop shortcut.
2. Select PowerPoint / live keys; tick Allow slide keys this session.
3. Open the test deck in slideshow mode, preferably away from its endpoints.
4. Click Start session and focus the slideshow during the eight-second countdown.
5. With your right hand open, show each side clearly. Palm to back sends NEXT;
   back to palm sends PREVIOUS. The initial side establishes the starting point.
6. A held fist for one second pauses/resumes; Victory for two seconds exits the
   slideshow and pauses control. Stop session releases the camera.

The 640 x 480 frame normalization remains in place. Palm/back thresholds and
algorithm behavior were not changed in this cleanup. Recording is optional via
the existing controls and stays beneath recordings/.

Former root-level Run Original GestureDeck.cmd and Undo Learned Swipe Changes.cmd
were moved into backups/20261119-before-palm-turn-only/archived_launchers/ to keep
them out of normal use. They are historical artifacts, not supported launchers
from their archived locations. Models, recordings, baseline/learned source and
previous milestone packages are preserved. Their old results do not become
palm/back accuracy results merely because the presentation mode changed.
