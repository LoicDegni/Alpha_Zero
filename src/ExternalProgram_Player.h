#pragma once

#include <iostream>
#include <cstdio>
#include <csignal>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <stdexcept>
#include <sstream>


#include "Hex_Environement.h"


class ExternalProgram : public Player_Interface {
    char player;
    pid_t child_pid = -1;
    int pipe_to_child[2] = {-1, -1};    // parent écrit [1], enfant lit [0]
    int pipe_from_child[2] = {-1, -1};  // enfant écrit [1], parent lit [0]

public:
    ExternalProgram(const std::string& path, char player, int taille) : player(player) {
        assert(player == 'X' || player == 'O');

        if (pipe(pipe_to_child) == -1 || pipe(pipe_from_child) == -1) {
            throw std::runtime_error("Échec de création des pipes");
        }

        child_pid = fork();

        if (child_pid == -1) {
            close(pipe_to_child[0]); close(pipe_to_child[1]);
            close(pipe_from_child[0]); close(pipe_from_child[1]);
            throw std::runtime_error("Échec du fork");
        }

        if (child_pid == 0) {
            // Processus enfant : fermer les extrémités inutilisées
            close(pipe_to_child[1]);
            close(pipe_from_child[0]);

            // Rediriger stdin/stdout vers les pipes, stderr vers /dev/null
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull == -1 ||
                dup2(pipe_to_child[0], STDIN_FILENO) == -1 ||       // Quand l'enfant lit stdin, il lit depuis pipe_to_child[0]
                dup2(pipe_from_child[1], STDOUT_FILENO) == -1 ||    // Quand l'enfant écrit sur stdout, il écrit dans pipe_from_child[1]
                dup2(devnull, STDERR_FILENO) == -1) {               // Rediriger stderr vers /dev/null
                perror("dup2");
                _exit(1);
            }
            close(devnull);

            // Fermer les FDs originaux après dup2
            close(pipe_to_child[0]);
            close(pipe_from_child[1]);

            // Remplacer le processus par le programme externe
            std::string player_str(1, player);
            std::string taille_str = std::to_string(taille);
            std::string player_arg = "-" + player_str;
            execlp(path.c_str(), path.c_str(),
                   player_arg.c_str(), "IA", "--size", taille_str.c_str(), "--noGUI", nullptr);

            // Si execlp échoue
            perror("execlp");
            _exit(1);
        }

        // Processus parent : fermer les extrémités inutilisées
        close(pipe_to_child[0]);
        close(pipe_from_child[1]);
    }

    ~ExternalProgram() {
        // Fermer les pipes
        close(pipe_to_child[1]);
        close(pipe_from_child[0]);

        // Attendre la fin du processus enfant pour éviter un zombie
        if (child_pid > 0) {
            waitpid(child_pid, nullptr, 0);
        }
    }

    void otherPlayerMove(int row, int col) override {
        std::string message = std::to_string(row) + " " + std::to_string(col) + "\n";
        write(pipe_to_child[1], message.c_str(), message.length());
    }

    std::tuple<int, int> getMove(Hex_Environement& hex) override {
        std::string line;
        char c;

        // Lire caractère par caractère jusqu'à '\n' pour garantir une ligne complète
        while (true) {
            ssize_t bytes_read = read(pipe_from_child[0], &c, 1);
            if (bytes_read <= 0) {
                // Pipe fermé ou erreur (l'enfant a probablement crashé)
                return {-1, -1};
            }
            if (c == '\n') break;
            line += c;
        }

        std::stringstream ss(line);
        int row, col;
        if (ss >> row >> col) {
            std::cerr << row << " " << col << std::endl;
            return {row, col};
        }

        return {-1, -1};
    }
};



