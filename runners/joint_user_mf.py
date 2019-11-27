from data.training import cold_start
from models.joint_user_mf import JointUserMF

if __name__ == '__main__':
    u_r_map = cold_start(
        from_path='../data/mindreader/user_ratings_map.json',
        conversion_map={
            -1: 1,
            0: None,  # Ignore don't know ratings
            1: 5
        },
        split_ratio=[75, 25]
    )

    n_iter = 100
    k = 10
    lr = 0.001

    model = JointUserMF()