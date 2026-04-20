/**
 * INF4230 - INTELLIGENCE ARTIFICIELLE
 * UQAM | Faculte des sciences | Departement d'informatique
 * TP4 : AlphaZero pour le jeu de Hex
 * PRENOM, NOM : Kaikou Loic Degni
 * CODE PERMANENT : DEGK24059500
 * 
*/

#pragma once

#include <tuple>
#include <vector>
#include <memory>
#include <cassert>
#include <limits>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cstdio>  
#include <random>
#include <unordered_set>

#include "Hex_Environement.h"

class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;
    std::vector<bool> occupied;
    std::vector<char> ownership;

    int top_virtual;
    int bottom_virtual;
    int left_virtual;
    int right_virtual;
    int N;

public:
    UnionFind(int n) : N(n) {
        int size = N * N + 4; 
        parent.resize(size);
        occupied.resize(size);
        ownership.resize(size,'-');
        rank.resize(size, 0);

        for (int i = 0; i < size; i++) {
            parent[i] = i;
            occupied[i] = false;
        }
        top_virtual = N * N;
        bottom_virtual = N * N + 1;
        left_virtual = N * N + 2;
        right_virtual = N * N + 3;
    }

    int id(int r, int c) const {
        return r * N + c;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);

        if (ra == rb) return;

        if (rank[ra] < rank[rb]) {
            parent[ra] = rb;
        }
        else if (rank[ra] > rank[rb]) {
            parent[rb] = ra;
        }
        else {
            parent[rb] = ra;
            rank[ra]++;
        }
    }

    bool connected(int a, int b) {
        return find(a) == find(b);
    }

    void reset() {
        int size = parent.size();
        for (int i = 0; i < size; i++) {
            parent[i] = i;
            rank[i] = 0;
            occupied[i] = false;
            ownership[i] = '-';
        }
    }

    void applyMoveUF(int r, int c, char player) {
        int node = id(r, c);
        occupied[node] = true;
        ownership[node] = player;
        //(6 voisins)
        int dr[6] = {-1, -1, 0, 0, 1, 1};
        int dc[6] = {0, 1, -1, 1, -1, 0};

        for (int i = 0; i < 6; i++) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                if (occupied[id(nr, nc)] && ownership[id(nr, nc)] == player)
                    unite(node, id(nr, nc));
            }
        }
        // connexions aux au bord
        if (player == 'O') {
            if (r == 0) unite(node, top_virtual);
            if (r == N - 1) unite(node, bottom_virtual);
        }
        if (player == 'X') {
            if (c == 0) unite(node, left_virtual);
            if (c == N - 1) unite(node, right_virtual);
        }
    }

    bool hasWinner(char player) {
        if (player == 'O') {
            return connected(top_virtual, bottom_virtual);
        }
        else {
            return connected(left_virtual, right_virtual);
        }
    }
};

class IA_Player : public Player_Interface {
private:
    char _player;
    unsigned int _taille;
    unsigned int _time_limit_ms = 2000; // Par défaut, 2 secondes par coup
    std::vector< std::tuple<unsigned int, unsigned int, char> > _historique_coups;
    std::mt19937 _random_number_generator;
    UnionFind _uf;

    struct Node {
        Node* parent= nullptr;
        std::vector<Node*> children;
        int moveRow, moveCol;
        char playerJustMoved;
        int visits = 0;
        double wins = 0;
        //Rave
        int rave_visits = 0;
        double rave_wins = 0;

        std::vector<int> toVisit;
        std::vector<int> untriedMoves;
    };   
    Node* _root = nullptr;

//-------------------ALGO MCTS-------------------//
    Node* select(Node* node) {
        /**
         * La fonction selectionne le noeud le plus prometteur
         * parmis tous les enfants du noeud courant.
         * Stratégie:
         *  Upper Confidence Trees (UCT)
         *  RAVE (Rapid Action Value Estimation)
        */
        double C = 1.414;
        Node* best = nullptr;
        double bestValue = -1e9;

        for(auto child: node->children) {
            double exploitation_S_i = child->wins / (child->visits);
            double exploration_S_i = C * sqrt(log(node->visits) / (child->visits));
            //On previent le cas ou child->rave_visits = 0
            double rave_ratio = child->rave_wins/(child->rave_visits +1e-6);
            double w = ( child->rave_visits/(child->visits + child->rave_visits + 1e-6) );
            double score = ((1 - w)*exploitation_S_i) + (w * rave_ratio) + exploration_S_i; 
            if (score > bestValue) 
            {
                bestValue = score;
                best = child;
            }
        }
        _uf.applyMoveUF(best->moveRow, best->moveCol, best->playerJustMoved);
        return best;
    }

    Node* expand(Node* node) {
        /**
         * Fonction qui recoit un noeud courant, recupere un mouvement possible
         * du noeud et creer un noeud enfant avec le mouvement recuperé
         * 
         * Return:      Le noeud enfant
        */
        int moveID;
        Node* child = new Node();
        
        moveID = node->untriedMoves.back();        
        node->untriedMoves.pop_back();

        child->parent = node;
        child->moveRow = convertIDToCoordonate(moveID).first;
        child->moveCol = convertIDToCoordonate(moveID).second;
        child->playerJustMoved = (node->playerJustMoved == 'X') ? 'O' : 'X';

        child->toVisit = node->toVisit;

        //maj de child->tovisit
        auto it = std::find(child->toVisit.begin(), child->toVisit.end(),moveID);
        if (it != child->toVisit.end()) {
            std::swap(*it,child->toVisit.back());
            child->toVisit.pop_back();
        }
        child->untriedMoves = child->toVisit;
        node->children.push_back(child);

        // On met a jour la carte _uf[O(n)]
        _uf.applyMoveUF(child->moveRow, child->moveCol, child->playerJustMoved);
        return child;
    }

    char simulate(Node* node) {
        /**
         * La fonction simule toute la suite de la partie 
         * du noeud courant, mets à jour les variables
         * et retourne le gagnant.
        */
        std::vector<int> played_moves;
        char pl = node->playerJustMoved;

        if (node->toVisit.empty()) {
            return node->playerJustMoved;
        }
        simulateToTheEnd(pl,node->toVisit, played_moves);
        raveSimulationUpdate(node, played_moves, pl);
        return pl;
    }

    void backpropagate(Node* node, char winner) {
        /**
         * La fonction remonte l'arbre MCTS et mets à jour les noeuds.
        */
       while (node != nullptr) {
        node->visits++;

        if (node->playerJustMoved == winner){
            node->wins++;
        }
        node = node->parent;
       }
    }
//-------------------ALGO MCTS-------------------//
public:
    IA_Player(char player, unsigned int taille=10) : _player(player), _taille(taille), _random_number_generator(std::random_device{}()), _uf(taille) {
        assert(player == 'X' || player == 'O');
    }

    void otherPlayerMove(int row, int col) override {
        /**
         * Lorsque l'agent le coup du joueur, 
         * il met à jour sont historique interne et 
         * avance dans l'arbre avec l'etat courant. 
         * Si il ne trouve pas de noeud dans son arbre qui correspond
         * au coup joué par l'adversaire, il creer une nouvelle racine 
         * avec l'etat courant
        */
        _historique_coups.push_back({row, col, (_player == 'X') ? 'O' : 'X'});

        if(_root != nullptr) {
            for(auto child : _root->children) {
                if(child->moveRow == row && child->moveCol == col) {
                    _root = child;
                    _root->parent = nullptr;
                    // On met a jour la carte _uf[O(n)]
                    resetUFToNow();
                    return;
                }
            }
            _root = nullptr; 
            // On met a jour la carte _uf[O(n)]
            resetUFToNow();
        }
    }

    std::tuple<int, int> getMove(Hex_Environement& hex) override {
        /**
         * Fonction qui à partir d'un noeud courant, explore l'arbre
         * MCTS le plus souvent possible pendant 1.9 secondes. puis retourne
         * le noeud le plus prometteur.
        */
        auto start = std::chrono::steady_clock::now();
        auto deadline = start + std::chrono::milliseconds(_time_limit_ms);

        if(_root == nullptr) {
            _root = new Node();
            _root->playerJustMoved = (_player == 'X') ? 'O' : 'X';
            getAllMoves(hex);
        }

        while (std::chrono::steady_clock::now() < deadline) {
            Node* node = _root;
            // 1. Sélection
            while(node->untriedMoves.empty() && !node->children.empty())
                node = select(node);
            // 2. Expansion
            if(!node->untriedMoves.empty())
                node = expand(node);
            // 3. Simulation
            char winner;
            if (!_uf.hasWinner(node->playerJustMoved)) 
                winner = simulate(node);
            else
                winner = node->playerJustMoved;
            // 4. Rétropropagation
            backpropagate(node,winner);
            resetUFToNow();
        }
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cerr << "Temps MCTS: " << duration.count() << " ms\n";
        Node* best = FindBestChild(_root);
        _historique_coups.push_back({best->moveRow,  best->moveCol, _player});
        _root = best;
        _root->parent = nullptr;
        return {best->moveRow, best->moveCol};
    }

    void MCTS_TimeLimit(unsigned int time_limit_ms) {
        _time_limit_ms = time_limit_ms;
    }

private:
    void getAllMoves(Hex_Environement& hex) {
        /**
         * Fonction qui recupere tout les coups valides restant
         * dans la partie et les mets à jours au noeud racine.
        */
        for(unsigned int i=0; i < _taille; i++) {
            for(unsigned int j = 0; j< _taille; j++) {
                if(hex.isValidMove(i,j)) {
                    _root->toVisit.push_back(convertCoordonateToID(i,j));
                    _root->untriedMoves.push_back(convertCoordonateToID(i,j));
                }
            }
        }
    }

    void simulateToTheEnd(char& pl, std::vector<int>& available_moves, std::vector<int>& played_moves){
        /**
         * Fonction qui simule une partie jusqu'a ce qu'il y ai un gagnant. 
         * La structure unionFind(_uf) simule l'etat du jeu
        */
        do {
            pl = (pl == 'X') ? 'O' : 'X';
            std::uniform_int_distribution<int> uniform_moves_distribution(0, available_moves.size() -1);
            int random_index = uniform_moves_distribution(_random_number_generator);
            auto id = available_moves[random_index];
            auto move = convertIDToCoordonate(id);
            played_moves.push_back(id);
            _uf.applyMoveUF(move.first, move.second, pl);
        }while (!_uf.hasWinner(pl));

        if (!_uf.hasWinner('X') && !_uf.hasWinner('O')){
            std::cerr << "Erreur: available list est vide\n";
            std::exit(EXIT_FAILURE);
        }
    }

    Node* FindBestChild(Node* node) {
        /**
         * Retourne le noeud le plus prometteurs
         * 
         * Le noeud le plus visité. Lorsque plusieurs noeud
         * ont le même nombre de visite, on compare leurs 
         * nombre de victoire
        */
        Node* best = nullptr;
        int maxVisits = -1;
        double bestWinrate = -1.0;

        for (auto child : node->children) {
            if (child->visits > maxVisits) {
                maxVisits = child->visits;
                bestWinrate = child->wins / (child->visits + 1e-6);
                best = child;
            } 
            else if (child->visits == maxVisits) {
                double winrate = child->wins / (child->visits + 1e-6);
                if (winrate > bestWinrate) {
                    bestWinrate = winrate;
                    best = child;
                }
            }
        }
        return best;
    }

    int distanceToCenter(int r, int c, int N) {
        /**
         * Fonction calcul la distance de Manhattan 
         * d'une position au centre de la table de jeu
        */
        int center = N / 2;
        return std::abs(r - center) + std::abs(c - center);
    }

    int convertCoordonateToID(int r, int c) {
        /**
         * Fonction qui convertit une coordonne(r,c)
         * en un identifiant unique
        */
        return r * _taille + c;
    }

    std::pair<int, int> convertIDToCoordonate(int id) {
        /**
         * Fonction qui convertit un identifiant 
         * en sa coordonnée(r,c) d'origine
        */
       return {id / _taille, id % _taille};
    }

    void resetUFToNow(){
        /**
         * Fonction qui remet à zéro la structure unionFind
         * et ensuite la mets à jours avec l'historiques de
         * coups courant.
        */
        _uf.reset();
        for(const auto& [r,c,pl]: _historique_coups) {
            _uf.applyMoveUF(r,c,pl);
            }
    }

    void raveSimulationUpdate(Node* node, std::vector<int>& move_played_in_simulation, char winner) {
        /**
         * Fonction parcourt les coups possible du noeud courant 
         * et verifie si le même coup est joué à un moment ou un 
         * autre de la simulation et si le coup a mené à la victoire.
         * 
         * Si oui dans un ou l'autre des cas on rajoute de la valeurs au
         * coup pour la sélection
        */
        for (auto id : move_played_in_simulation) {
            auto move = convertIDToCoordonate(id);
            for (auto child : node->children) {
                if (child->moveRow == move.first && child->moveCol == move.second) {
                    child->rave_visits++;
                    if (child->playerJustMoved == winner) child->rave_wins++;
                }
            }
        }
    }
};
