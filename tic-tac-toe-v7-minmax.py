import cv2, numpy as np, time, random
from dataclasses import dataclass
from typing import List, Optional
from pydobot import Dobot

# ----------------- Dobot setup -----------------
DOBOT_PORT = "/dev/ttyACM0"   
TRAVEL_Z, HOVER_Z, PICK_Z = 23, -15, -44
HOME = (200, 0, TRAVEL_Z, 0)
#-------------------------------------------------
device = Dobot(port=DOBOT_PORT)
device.speed(150, 150)
device.suck(False)
device.move_to(*HOME)
#-------------------------------------------------
PICK_POINTYellow= (221, -100)     
PICK_POINTGreen = (259, -100)     
humanPickupYNegativ = 0
ComputerPickupYNegativ = 0
PickupYNegativDistanc = 20
CountStable = 150
#-------------------------------------------------
DESTS = {
    "0": (220, -42),"1": (220, -8),"2": (220, 23),
    "3": (253, -42),"4": (253, -8),"5": (253, 23),
    "6": (285, -42),"7": (285, -8),"8": (285, 23),
}
# ----------------- Camera -----------------
cap = cv2.VideoCapture(2)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------------- (HSV thresholds) -----------------
HSV_GREEN = ((35, 80, 60), (85, 255, 255)) # Green
HSV_YELLOW = ((20, 100, 100), (35, 255, 255)) # Yellow
MIN_RATIO = 0.06            
MORPH_K   = 3              
WARP_SIZE = 300            
# ================= PASS-BY-REFERENCE STYLE STATE =================
@dataclass
class CamState:
    labels: Optional[List[str]] = None      
    prev:   Optional[List[str]] = None     
    last_stable: Optional[List[str]] = None 
    frame:  Optional[np.ndarray] = None    
    show:   Optional[np.ndarray] = None    

# ----------------- Window/ -----------------
_windows_ready = False

def ensure_windows():
    global _windows_ready
    if not _windows_ready:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Board Read (warped)", cv2.WINDOW_NORMAL)
        _windows_ready = True

def show_frames(state: CamState):
    ensure_windows()

    if state.frame is None:
        cam_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(cam_img, "No frame...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    else:
        cam_img = state.frame

    if state.show is None:
        warp_img = np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)
        cv2.putText(warp_img, "Waiting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    else:
        warp_img = state.show

    cv2.imshow("Camera", cam_img)
    cv2.imshow("Board Read (warped)", warp_img)
    cv2.waitKey(1)  # critical for window event loop

# ----------------- Robot moves -----------------
def pick(x, y):
    time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0); 
    time.sleep(0.3)
    device.move_to(x, y, HOVER_Z,  0);
    time.sleep(0.3)
    device.move_to(x, y, PICK_Z,   0); 
    time.sleep(0.2)
    device.suck(True); time.sleep(0.35)
    device.move_to(x, y, HOVER_Z,  0);
    time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0);
    time.sleep(0.3)


def place(x, y):
    time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0);
    time.sleep(0.3)
    device.move_to(x, y, HOVER_Z,  0);
    time.sleep(0.3)
    device.move_to(x, y, PICK_Z,   0); 
    time.sleep(0.2)
    device.suck(False); time.sleep(0.3)
    device.move_to(x, y, HOVER_Z,  0); 
    time.sleep(0.3)
    device.move_to(x, y, TRAVEL_Z, 0); 
    time.sleep(0.3)

def winner(b):
    lines = [(0,1,2),(3,4,5),(6,7,8),
             (0,3,6),(1,4,7),(2,5,8),
             (0,4,8),(2,4,6)]
    for a,c,d in lines:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None

def is_draw(b):
    return all(x != " " for x in b) and winner(b) is None
#=====
def computer_move_minimax(b, comp):
    
    from functools import lru_cache
    human = "Y" if comp == "G" else "G"

    if b[4] == " " and b.count(" ") >= 8:
        return 4

  
    corners = [0, 2, 6, 8]
    if b.count(" ") >= 8:  # still early in game
        for c in corners:
            if b[c] == " ":
                return c

    # ---------- Helper functions ----------
    lines = [(0,1,2),(3,4,5),(6,7,8),
             (0,3,6),(1,4,7),(2,5,8),
             (0,4,8),(2,4,6)]

    def win_local(board):
        for a,b,c in lines:
            if board[a] != " " and board[a] == board[b] == board[c]:
                return board[a]
        return None

    def draw_local(board):
        return all(x != " " for x in board) and win_local(board) is None

    def immediate_winning_move(board, token):
        for i, v in enumerate(board):
            if v == " ":
                board[i] = token
                if win_local(board) == token:
                    board[i] = " "
                    return i
                board[i] = " "
        return None

    win_now = immediate_winning_move(b[:], comp)
    if win_now is not None:
        return win_now

    block_now = immediate_winning_move(b[:], human)
    if block_now is not None:
        return block_now


    @lru_cache(maxsize=None)
    def minimax(state_str, turn):
        board = list(state_str)
        w = win_local(board)
        if w == comp: return (10, None)
        if w == human: return (-10, None)
        if draw_local(board): return (0, None)

        empties = [i for i,v in enumerate(board) if v == " "]
        if not empties: return (0, None)

        if turn == comp:
            best, best_i = -999, None
            for i in empties:
                board[i] = comp
                score, _ = minimax("".join(board), human)
                board[i] = " "
                if score > best:
                    best, best_i = score, i
            return (best - 1, best_i)
        else:
            worst, worst_i = 999, None
            for i in empties:
                board[i] = human
                score, _ = minimax("".join(board), comp)
                board[i] = " "
                if score < worst:
                    worst, worst_i = score, i
            return (worst + 1, worst_i)

    score, idx = minimax("".join(b), comp)
    return idx if idx is not None else random.choice([i for i,v in enumerate(b) if v == " "])

_calib_pts = []

def _on_mouse(event, x, y, flags, param):
    global _calib_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(_calib_pts) < 4:
        _calib_pts.append((x, y))

def load_calibration():
    """Load saved calibration matrix if available."""
    try:
        M = np.load("calibration_matrix.npy")
        print("[Calib] Loaded existing calibration.")
        return M
    except Exception:
        print("[Calib] No saved file found.")
        return None


def calibrate_board(cap):
    """Click 4 corners: TL, TR, BR, BL. Enter=OK | c=clear | q=quit"""
    global _calib_pts
    _calib_pts = []
    win = "Board Calib"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)
    print("[Calib] Click 4 corners BR,BL,Tl,TR then press Enter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        for i, p in enumerate(_calib_pts):
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            _calib_pts = []
        elif key == ord('q'):
            cv2.destroyWindow(win)
            return None
        elif key in (13, 10):  # Enter
            if len(_calib_pts) == 4:
                src = np.array(_calib_pts, dtype=np.float32)
                dst = np.array([[0, 0], [WARP_SIZE, 0],
                                [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src, dst)
                np.save("calibration_matrix.npy", M)  
                print("[Calib] Saved calibration to calibration_matrix.npy")
                cv2.destroyWindow(win)
                return M
            else:
                print("[Calib] Please click 4 points!")
def _mask_color(hsv, low, high):
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))


def classify_cell(cell_bgr):
    """cell_bgr: 100x100 BGR → returns 'Y', 'G' or ' '"""
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    HSV_GREEN = ((35, 80, 60), (85, 255, 255)) # Green
    HSV_YELLOW = ((20, 100, 100), (35, 255, 255)) # Yellow

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    
    yellow = _mask_color(hsv, *HSV_YELLOW)
    yellow = cv2.morphologyEx(yellow,  cv2.MORPH_OPEN, k)

    green = _mask_color(hsv, *HSV_GREEN)
    green= cv2.morphologyEx(green,  cv2.MORPH_OPEN, k)
    
    s_mask = (hsv[:,:,1] >= 60).astype(np.uint8)*255
    v_mask = (hsv[:,:,2] >= 60).astype(np.uint8)*255
    valid  = cv2.bitwise_and(s_mask, v_mask)

    g_ratio = green.sum() / 255.0 / green.size
    y_ratio = yellow.sum() / 255.0 / yellow.size
    if g_ratio < MIN_RATIO and y_ratio < MIN_RATIO:
       return " "
    return "G" if g_ratio >= y_ratio else "Y"


def split_cells(board_bgr):
    
    cells = []
    step = WARP_SIZE // 3
    pad  = 6
    for r in range(3):
        for c in range(3):
            y1, y2 = r*step+pad, (r+1)*step-pad
            x1, x2 = c*step+pad, (c+1)*step-pad
            cells.append(board_bgr[y1:y2, x1:x2].copy())
    return cells


def read_board_once(cap, M):
   
    ok, frame = cap.read()
    if not ok:
        return None, None, None

    warped = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))
    cells  = split_cells(warped)
    labels = [classify_cell(c) for c in cells]

    show = warped.copy()
    step = WARP_SIZE // 3
    for idx, lab in enumerate(labels):
        r, c = divmod(idx, 3)
        cx, cy = c*step + step//2, r*step + step//2
        color = (0,255,0) if lab=="G" else (0,255,255) if lab=="Y" else (200,200,200)
        cv2.putText(show, (lab if lab!=" " else "0"), (cx-10, cy+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.rectangle(show, (c*step, r*step), ((c+1)*step, (r+1)*step), (80,80,80), 1)
    return labels, frame, show


def diff_new_tokens(prev_labels, curr_labels):
    diffs = []
    n = min(len(prev_labels or []), len(curr_labels or []))
    if not prev_labels:
        for i in range(n):
            if curr_labels[i] in ("Y","G"):
                diffs.append((i, curr_labels[i]))
        return diffs
    for i in range(n):
        if prev_labels[i] == " " and curr_labels[i] in ("Y","G"):
            diffs.append((i, curr_labels[i]))
    if len(curr_labels) > n:
        for i in range(n, len(curr_labels)):
            if curr_labels[i] in ("Y","G"):
                diffs.append((i, curr_labels[i]))
    return diffs

# ----------------- Pass-by-ref style human interactions -----------------
def SelectColorHuman(b, cap, M, state: CamState, CountStable=100):
    """Human drops the very first cube to declare color; updates `state` in-place."""
    Flag_Stable = 0
    while True:
        labels_now, frame, show_img = read_board_once(cap, M)
        state.frame, state.show = frame, show_img
        show_frames(state)

        if labels_now is None:
            continue

        diffs = diff_new_tokens(state.prev, labels_now)
        Flag_Stable = 0 if diffs else (Flag_Stable + 1)
        state.labels = labels_now[:]
        state.prev   = labels_now[:]

        if Flag_Stable >= CountStable:
            for i in range(len(labels_now)):
                if labels_now[i] != " " and b[i] == " ":
                    choice = labels_now[i]
                    state.last_stable = labels_now[:] 
                    return i, M, choice

#--------------------------------------------------------------------------------------
def human_move_gui(b, cap, M, state: CamState, CountStable=100):
   
    Flag_Stable = 0
    while True:
        labels_now, frame, show_img = read_board_once(cap, M)
        #print(labels_now)
        state.frame, state.show = frame, show_img
        if state.show is not None:
            cv2.putText(state.show, "Your move (place a cube) | c=recalib, q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        show_frames(state)

        if labels_now is None:
            continue

        diffs = diff_new_tokens(state.prev, labels_now)
        Flag_Stable = 0 if diffs else (Flag_Stable + 1)
        state.labels = labels_now[:]
        state.prev   = labels_now[:]

        if Flag_Stable >= CountStable:
           
            if state.last_stable is None:
                state.last_stable = labels_now[:]
                continue
            diffs2 = diff_new_tokens(state.last_stable, labels_now)
            if diffs2:
                i, color = diffs2[0]
                if b[i] == " ":
                    DefCount=len(diffs2) 
                    
                    state.last_stable = labels_now[:]
                    return i,color, M,DefCount
                else:
                    cv2.displayOverlay("Board Read (warped)",
                                       f"Cell {i+1} is occupied. Pick another.", 1000)

# ----------------- MAIN -----------------
def main():
    global humanPickupYNegativ, ComputerPickupYNegativ

   
    for _ in range(10):
        cap.read()
        cv2.waitKey(1)

    
    b = [" "] * 9
    state = CamState()
    
    first = input("Who goes first? (H for Human, R for Robot, or D for Double-Human mode) ").strip().upper()
    turn = "human" if first == "H" else "ROBOT"
    
    if first=="H":
         turn="human"
    else:
        if first=="R" :     
           turn="ROBOT"   

    M = load_calibration()
    if M is None:
        M = calibrate_board(cap)
        if M is None:
            print("Calibration canceled.")
            return
    labels0, frame0, show0 = read_board_once(cap, M)
    state.labels = labels0[:] if labels0 else None
    state.prev   = labels0[:] if labels0 else None
    state.last_stable = labels0[:] if labels0 else None
    state.frame  = frame0
    state.show   = show0
    show_frames(state)

       
    human, comp ,human2= "Y", "G", "G"

    if first == "R":
        human, comp = "Y", "G"
        PICK_HUMAN = PICK_POINTYellow
        PICK_COMP  = PICK_POINTGreen
    elif first=="H":
        
        print("Please place the first cube in the desired color.")
        i, M, choice = SelectColorHuman(b, cap, M, state, CountStable=CountStable)
       
        b[i] = choice
        turn = "ROBOT"
        if choice == "Y":
            human, comp = "Y", "G"
            PICK_HUMAN = PICK_POINTYellow
            PICK_COMP  = PICK_POINTGreen
        else:
            human, comp = "G", "Y"
            PICK_HUMAN = PICK_POINTGreen
            PICK_COMP  = PICK_POINTYellow
    elif first=="D":
        print("Please place the first cube* in the desired color.")
        i, M, choice = SelectColorHuman(b, cap, M, state, CountStable=CountStable)
       
        b[i] = choice
        
        turn = "HUMAN2"
      
        if choice == "Y":
            human, human2 = "Y", "G"
           
        else:
            human, human2 = "G", "Y"
            
    humanPickupYNegativ = 0
    ComputerPickupYNegativ = 0
    print(turn)
    while True:
        if turn == "human":
            print("Human is playing...Insert youe cube")
            i,color2, M ,DefCount= human_move_gui(b, cap, M, state, CountStable=CountStable)
            b[i] = human
            #print(DefCount)
            if  DefCount>=2 or color2!=human:
                for kk in range(5):
                    print("⚠️ Cheating detected! You placed more than one cube OR Wrong color detected! Game terminated")
                break
        elif turn == "ROBOT":
            print("ROBOT is playing...")
           
            i = computer_move_minimax(b, comp)
            
            b[i] = comp
            state.last_stable[i]=comp

            x, y = PICK_COMP
            y = y - ComputerPickupYNegativ
            ComputerPickupYNegativ += PickupYNegativDistanc
            pick(x, y)

            coord = DESTS[str(i)]
            print("Robot place ")
            place(*coord)
         
            
        elif turn == "HUMAN2":
            print("Human 2 is playing...Insert your cube")
            i,color2, M ,DefCount= human_move_gui(b, cap, M, state, CountStable=CountStable)
            b[i] = human2
           
            if  DefCount>=2 or color2!=human2:
                for kk in range(5):
                    print("⚠️ Cheating detected! You placed more than one cube OR Wrong color detected! Game terminated")
                break


      
        device.move_to(*HOME)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            M = calibrate_board(cap)
        elif k == ord('q'):
            break

        w = winner(b)
        
        if w or is_draw(b):
            if w==human2 and first=="D" :
                 print("🏆 Congratulations! You outsmarted the Human2 — You win! 🎉")
            elif w == human :
                print("🏆 Congratulations! Human is a winner — You win! 🎉")
            elif w == comp:
                time.sleep(5)
                print("🤖 The robot prevails! Better luck next time, human.")
            else:
                time.sleep(5)
                print("🤝 Draw! Seems we’re equally smart this time.")
            break
        
        if first != "D":
          turn = "ROBOT" if turn == "human" else "human"
        else :
           turn = "HUMAN2" if turn == "human" else "human"  

        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            M = calibrate_board(cap)
        elif k == ord('q'):
            break

       

if __name__ == "__main__":
    try:
        device.suck(False)
        main()
       # show_frames(state)
        input()
    finally:
        cap.release(); cv2.destroyAllWindows()
        device.suck(False)
        device.move_to(*HOME)
        device.close()
