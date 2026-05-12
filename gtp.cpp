#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include "boardstate.hpp"
#include "zobrist.hpp"
#include "mcts.hpp"

// convert GTP coordinate like "D4" to pos int
int gtpToPos(const std::string& coord) {
    if (coord == "pass" || coord == "PASS") return -1;
    char col = toupper(coord[0]);
    if (col >= 'I') col--;  // GTP skips 'I'
    int c = col - 'A';
    int r = 19 - std::stoi(coord.substr(1));
    return r * 19 + c;
}

std::string posToGtp(int pos) {
    if (pos == -1) return "pass";
    int r = pos / 19;
    int c = pos % 19;
    char col = 'A' + c;
    if (col >= 'I') col++;
    return std::string(1, col) + std::to_string(19 - r);
}

int main() {
    Zobrist::init();
    MCTS mcts(1.41, 6.5f, "gonet.pt");
    boardstate b{};
    std::unordered_set<uint64_t> history;
    int cmdId = -1;

    auto ok  = [&](const std::string& msg) {
        std::cout << (cmdId >= 0 ? "=" + std::to_string(cmdId) : "=")
                  << " " << msg << "\n\n" << std::flush;
    };
    auto err = [&](const std::string& msg) {
        std::cout << (cmdId >= 0 ? "?" + std::to_string(cmdId) : "?")
                  << " " << msg << "\n\n" << std::flush;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string token;
        ss >> token;

        // check for optional command id
        cmdId = -1;
        std::string cmd = token;
        if (isdigit(token[0])) {
            cmdId = std::stoi(token);
            ss >> cmd;
        }

        if (cmd == "name")         ok("GoBot");
        else if (cmd == "version") ok("1.0");
        else if (cmd == "protocol_version") ok("2");
        else if (cmd == "boardsize") ok("");
        else if (cmd == "clear_board") {
            b = boardstate{};
            history.clear();
            ok("");
        }
        else if (cmd == "komi") ok("");
        else if (cmd == "play") {
            std::string color, coord;
            ss >> color >> coord;
            int pos = gtpToPos(coord);
            history.insert(b.zobristHash);
            if (pos == -1) b = b.makePass();
            else b = b.makeMove(pos);
            ok("");
        }
        else if (cmd == "genmove") {
            int move = mcts.getBestMove(b, history, 200);
            history.insert(b.zobristHash);
            if (move == -1) b = b.makePass();
            else b = b.makeMove(move);
            ok(posToGtp(move));
        }
        else if (cmd == "quit") { ok(""); break; }
        else err("unknown command");
    }
    return 0;
}
