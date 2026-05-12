import pygame
import subprocess
import sys
import threading
import time

# ── GTP engine wrapper ────────────────────────────────────────────────────────

class GTPEngine:
    def __init__(self, path):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            if line == "":
                break
            lines.append(line)
        response = "\n".join(lines)
        if response.startswith("="):
            return response[1:].strip()
        raise RuntimeError(f"GTP error: {response}")

    def play(self, color, pos):
        self.send(f"play {color} {pos}")

    def genmove(self, color):
        return self.send(f"genmove {color}")

    def clear(self):
        self.send("clear_board")

    def quit(self):
        try: self.send("quit")
        except: pass
        self.proc.terminate()

# ── coordinate helpers ────────────────────────────────────────────────────────

COLS = "ABCDEFGHJKLMNOPQRST"  # GTP skips I

def pixel_to_board(x, y, margin, cell):
    col = round((x - margin) / cell)
    row = round((y - margin) / cell)
    return row, col

def board_to_pixel(row, col, margin, cell):
    return margin + col * cell, margin + row * cell

def pos_to_gtp(row, col):
    return COLS[col] + str(19 - row)

def gtp_to_rc(gtp):
    if gtp.lower() == "pass": return None
    col = COLS.index(gtp[0].upper())
    row = 19 - int(gtp[1:])
    return row, col

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.display.set_caption("Go")

    SIZE       = 700
    MARGIN     = 50
    CELL       = (SIZE - 2 * MARGIN) // 18
    STONE_R    = int(CELL * 0.47)
    INFO_W     = 220

    # warm wood palette
    BG         = (210, 170, 100)
    LINE       = ( 90,  60,  20)
    BLACK_S    = ( 20,  20,  20)
    WHITE_S    = (245, 245, 240)
    HIGHLIGHT  = (220,  80,  40)
    PANEL_BG   = ( 50,  35,  20)
    TEXT_COLOR = (230, 210, 170)
    STAR_COLOR = ( 80,  50,  15)

    screen = pygame.display.set_mode((SIZE + INFO_W, SIZE))
    font_large = pygame.font.SysFont("Georgia", 22, bold=True)
    font_small = pygame.font.SysFont("Georgia", 16)

    engine = GTPEngine("./gobot")
    engine.clear()

    board      = [[0]*19 for _ in range(19)]  # 0=empty 1=black 2=white
    turn       = 1   # 1=black(human) 2=white(bot)
    last_move  = None
    status_msg = "Your turn (Black)"
    game_over  = False
    bot_thinking = False
    passes     = 0

    star_points = [(3,3),(3,9),(3,15),(9,3),(9,9),(9,15),(15,3),(15,9),(15,15)]

    def draw():
        screen.fill(BG)

        # board lines
        for i in range(19):
            x1, y1 = board_to_pixel(i, 0,  MARGIN, CELL)
            x2, y2 = board_to_pixel(i, 18, MARGIN, CELL)
            pygame.draw.line(screen, LINE, (x1, y1), (x2, y2), 1)
            x1, y1 = board_to_pixel(0,  i, MARGIN, CELL)
            x2, y2 = board_to_pixel(18, i, MARGIN, CELL)
            pygame.draw.line(screen, LINE, (x1, y1), (x2, y2), 1)

        # star points
        for r, c in star_points:
            px, py = board_to_pixel(r, c, MARGIN, CELL)
            pygame.draw.circle(screen, STAR_COLOR, (px, py), 5)

        # stones
        for r in range(19):
            for c in range(19):
                if board[r][c] == 0: continue
                px, py = board_to_pixel(r, c, MARGIN, CELL)
                color = BLACK_S if board[r][c] == 1 else WHITE_S
                shadow = (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40))
                pygame.draw.circle(screen, shadow, (px+2, py+2), STONE_R)
                pygame.draw.circle(screen, color,  (px,   py),   STONE_R)
                if board[r][c] == 2:
                    pygame.draw.circle(screen, (180,180,175), (px, py), STONE_R, 1)

        # last move marker
        if last_move:
            r, c = last_move
            px, py = board_to_pixel(r, c, MARGIN, CELL)
            col = HIGHLIGHT
            pygame.draw.circle(screen, col, (px, py), STONE_R // 3)

        # info panel
        panel_rect = pygame.Rect(SIZE, 0, INFO_W, SIZE)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)

        y = 30
        title = font_large.render("GO", True, TEXT_COLOR)
        screen.blit(title, (SIZE + INFO_W//2 - title.get_width()//2, y))
        y += 50

        turn_str = "Black (You)" if turn == 1 else "White (Bot)"
        t = font_small.render(f"Turn: {turn_str}", True, TEXT_COLOR)
        screen.blit(t, (SIZE + 15, y)); y += 30

        msg_lines = [status_msg[i:i+18] for i in range(0, len(status_msg), 18)]
        for line in msg_lines:
            s = font_small.render(line, True, TEXT_COLOR)
            screen.blit(s, (SIZE + 15, y)); y += 22

        y += 20
        pass_btn = pygame.Rect(SIZE + 20, y, INFO_W - 40, 38)
        pygame.draw.rect(screen, (80, 55, 30), pass_btn, border_radius=6)
        pt = font_small.render("Pass", True, TEXT_COLOR)
        screen.blit(pt, (pass_btn.centerx - pt.get_width()//2,
                         pass_btn.centery - pt.get_height()//2))

        y += 55
        new_btn = pygame.Rect(SIZE + 20, y, INFO_W - 40, 38)
        pygame.draw.rect(screen, (60, 40, 20), new_btn, border_radius=6)
        nt = font_small.render("New Game", True, TEXT_COLOR)
        screen.blit(nt, (new_btn.centerx - nt.get_width()//2,
                         new_btn.centery - nt.get_height()//2))

        pygame.display.flip()
        return pass_btn, new_btn

    def bot_move_thread():
        nonlocal turn, last_move, status_msg, game_over, bot_thinking, passes
        bot_thinking = True
        status_msg = "Bot thinking..."
        try:
            gtp = engine.genmove("white")
            if gtp.lower() == "pass":
                passes += 1
                last_move = None
                status_msg = "Bot passed"
                if passes >= 2:
                    game_over = True
                    status_msg = "Game over!"
            else:
                passes = 0
                rc = gtp_to_rc(gtp)
                if rc:
                    r, c = rc
                    board[r][c] = 2
                    last_move = (r, c)
            turn = 1
            if not game_over:
                status_msg = "Your turn (Black)"
        except Exception as e:
            status_msg = f"Error: {e}"
        bot_thinking = False

    def new_game():
        nonlocal board, turn, last_move, status_msg, game_over, passes
        board = [[0]*19 for _ in range(19)]
        turn = 1
        last_move = None
        status_msg = "Your turn (Black)"
        game_over = False
        passes = 0
        engine.clear()

    clock = pygame.time.Clock()
    running = True
    while running:
        pass_btn, new_btn = draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not bot_thinking:
                mx, my = event.pos

                if new_btn.collidepoint(mx, my):
                    new_game()
                    continue

                if pass_btn.collidepoint(mx, my) and not game_over and turn == 1:
                    passes += 1
                    engine.play("black", "pass")
                    last_move = None
                    if passes >= 2:
                        game_over = True
                        status_msg = "Game over!"
                    else:
                        turn = 2
                        t = threading.Thread(target=bot_move_thread, daemon=True)
                        t.start()
                    continue

                if game_over or turn != 1 or mx >= SIZE:
                    continue

                r, c = pixel_to_board(mx, my, MARGIN, CELL)
                if 0 <= r < 19 and 0 <= c < 19 and board[r][c] == 0:
                    gtp = pos_to_gtp(r, c)
                    try:
                        engine.play("black", gtp)
                        board[r][c] = 1
                        last_move = (r, c)
                        passes = 0
                        turn = 2
                        status_msg = "Bot thinking..."
                        t = threading.Thread(target=bot_move_thread, daemon=True)
                        t.start()
                    except Exception as e:
                        status_msg = f"Illegal move"

        clock.tick(30)

    engine.quit()
    pygame.quit()

if __name__ == "__main__":
    main()
