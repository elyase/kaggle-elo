import chess.pgn
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, Counter
import chess
from chess import QUEEN
import pickle
import sys
import numpy as np
import pandas as pd


def get_games(filename="/Users/yaser/Copy/Code/kaggle/chess_elo/data.pgn",
              n_games=sys.maxsize):
    with open(filename) as pgn:
        game = chess.pgn.read_game(pgn)
        cnt = 0
        while game and cnt < n_games:
            cnt += 1
            yield game
            game = chess.pgn.read_game(pgn)


def get_ecos():
    with open('ecos.pickle', 'rb') as f:
        return pickle.load(f)


def extract_game_features(game):
    """Extract features for a given game"""

    first_check = True
    first_queen_move = True

    features = defaultdict(int)
    node = game

    while node.variations:  # and node.board().fullmove_number < cut_off:
        move = node.variation(0).move

        board = node.board()

        # print(board.fullmove_number, move)
        moved_piece = board.piece_type_at(move.from_square)
        captured_piece = board.piece_type_at(move.to_square)

        if moved_piece == QUEEN and first_queen_move:
            features['queen_moved_at'] = board.fullmove_number
            first_queen_move = False

        if captured_piece == QUEEN:
            features['queen_changed_at'] = board.fullmove_number

        # if captured_piece:
            # print('Capture', board.fullmove_number, move, moved_piece,captured_piece)
            # captures[]
            # if board.fullmove_number == 10:
            #    features['captures_5']
        if move.promotion:
            features['promotion'] += 1
        if board.is_check():
            features['total_checks'] += 1
            if first_check:
                features['first_check_at'] = board.fullmove_number
                first_check = False
        # castling
        uci_repr = move.uci()
        if uci_repr == 'e1g1':
            features['white_king_castle'] = board.fullmove_number
        elif uci_repr == 'e1c1':
            features['white_queen_castle'] = board.fullmove_number
        elif uci_repr == 'e8g8':
            features['black_king_castle'] = board.fullmove_number
        elif uci_repr == 'e8c8':
            features['black_queen_castle'] = board.fullmove_number

        node = node.variation(0)
    if board.is_checkmate():
        features['is_checkmate'] += 1
    if board.is_stalemate():
        features['is_stalemate'] += 1
    if board.is_insufficient_material():
        features['insufficient_material'] += 1
    if board.can_claim_draw():
        features['can_claim_draw'] += 1
    features['total_moves'] = board.fullmove_number

    # Pieces at the end of the game
    piece_placement = board.fen().split()[0]
    end_pieces = Counter(x for x in piece_placement if x.isalpha())

    # count number of piece at end position
    features.update({'end_' + piece: cnt
                     for piece, cnt in end_pieces.items()})
    return features


def games_features(games, return_names=False):
    l = []
    for game in games:
        features = extract_game_features(game)
        l.append(features)

    vec = DictVectorizer()
    X = vec.fit_transform(l)
    if return_names:
        return X, vec.get_feature_names()
    else:
        return X


def extract_score_features(scores):
    features = dict()
    if scores.size == 0:
        return features
    scores = np.r_[0, scores]
    abs_scores = np.abs(scores)
    diffs = np.diff(scores)
    white_diffs = diffs[::2]
    black_diffs = diffs[1::2]
    subset_names = ['diffs', 'white_diffs', 'black_diffs']
    subsets = [diffs, white_diffs, black_diffs]
    stats = [np.min, np.max, np.mean, lambda x: np.median(np.abs(x))]
    stat_names = ['min', 'max', 'mean', 'median_abs']
    for subset, subset_name in zip(subsets, subset_names):
        for stat, stat_name in zip(stats, stat_names):
            features[stat_name + '_' + subset_name] = stat(subset)
            # np.hi
    features['advantage120_idx'] = np.argmax(abs_scores > 120) or len(scores)
    features['advantage70_idx'] = np.argmax(abs_scores > 70) or len(scores)
    return features


def score_features(stockfish_scores):
    l = stockfish_scores.MoveScores.apply(extract_score_features).tolist()
    vec = DictVectorizer()
    X = vec.fit_transform(l)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    return df
