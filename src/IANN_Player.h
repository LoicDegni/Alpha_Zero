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


#include "HexCNN.h"
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


class IANN_Player : public Player_Interface {
    HexCNN _net;
    char _player;
    unsigned int _taille;
    unsigned int _time_limit_ms = 2000; // Par défaut, 2 secondes par coup
    std::vector< std::tuple<unsigned int, unsigned int, char> > _historique_coups;
    float _C_puct;
    UnionFind _uf;
    std::mt19937 _random_number_generator;
    std::mt19937 _rng;

    struct Node {
        Node* parent= nullptr;
        std::vector<Node*> children;
        int moveRow, moveCol;
        char playerJustMoved;
        int visits = 0;
        double wins = 0;

        std::vector<float> _politique;
        std::vector<int> toVisit;
        std::vector<int> untriedMoves;
    };
    Node* _root = nullptr;

    std::vector<float> get_dirichlet_noise(int taille_tableau, float alpha = 0.3f ) {
        // Génération du bruit de Dirichlet via des lois Gamma
        std::gamma_distribution<float> gamma(alpha, 1.0f);
        std::vector<float> noise(taille_tableau);
        float sum = 0.0f;
        for (size_t k = 0; k < noise.size(); ++k) {
            noise[k] = gamma(_rng);
            sum += noise[k];
        }
        
        // Normalisation pour que la somme soit égale à 1
        for (size_t k = 0; k < noise.size(); ++k) {
            noise[k] /= sum;
        }

        return noise;
    }

//-------------------ALGO MCTS-------------------//
    Node* select(Node* node) {
        /**
         * La fonction selectionne le noeud le plus prometteur
         * parmis tous les enfants du noeud courant.
         * Stratégie:
         *  Upper Confidence Trees (UCT)
        */
        double C = 1.414;
        Node* best = nullptr;
        double bestValue = -1e9;

        for(auto child: node->children) {
            double exploitation_S_i = child->wins / (child->visits);
            double exploration_S_i = C * sqrt(log(node->visits) / (child->visits));
            double score =  exploration_S_i + exploitation_S_i 
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
        //child->_politique = _net.evaluateState();
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
// Le constructeur par défaut avec HexCNN en paramètre
    IANN_Player(const HexCNN& net, char player, unsigned int taille=10, float Cpuct=2.5f) 
        : _net(net), _player(player), _taille(taille), _C_puct(Cpuct), _rng(std::random_device{}()), _random_number_generator(std::random_device{}()), _uf(taille) {
        assert(player == 'X' || player == 'O');
    }

    void otherPlayerMove(int row, int col) override {
        // l'autre joueur à joué (row, col)
    }

    std::tuple<int, int> getMove(Hex_Environement& hex) override {
        int row, col; // TODO TP4 : choisir le coups row, col a jouer

        ///// Exemple d'un choix aléatoire ////////////
        do {
            row = rand()%_taille; 	// Choix aléatoire
            col = rand()%_taille; 	// Choix aléatoire
        } while( hex.isValidMove(row, col) == false );
        //////////////////////////////////////////////

        return {row, col};
    }

    void MCTS_TimeLimit(unsigned int time_limit_ms) {
        _time_limit_ms = time_limit_ms;
    }

    void unactivateValueHead() {
        // TODO TP4 : désactiver la tête de valeur du CNN
    }

    void enableDataCollection(std::vector<TrainingExample>* examples) {
        // TODO TP4 : activer la collecte de données pour l'entraînement du CNN
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


};
