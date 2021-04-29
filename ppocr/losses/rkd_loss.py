import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# reference: https://github.com/lenscloth/RKD/blob/master/metric/loss.py


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(axis=1)
    prod = paddle.mm(e, e.t())
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clip(
        min=eps)

    if not squared:
        res = res.sqrt()

    return res


class RKdAngle(nn.Layer):
    def __init__(self):
        super(RKdAngle, self).__init__()

    def forward(self, student, teacher):
        # N x C
        # N x N x C
        # torch contains no_grad, however, we may also need to update teacher sometimes

        # reshape for feature map distillation
        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, axis=2)
        t_angle = paddle.bmm(norm_td, norm_td.transpose([0, 2, 1])).reshape(
            [-1, 1])

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, axis=2)
        s_angle = paddle.bmm(norm_sd, norm_sd.transpose([0, 2, 1])).reshape(
            [-1, 1])
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Layer):
    def __init__(self, eps=1e-12):
        super(RkdDistance, self).__init__()
        self.eps = eps

    def forward(self, student, teacher):
        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        t_d = pdist(teacher, squared=False)
        mean_td = t_d.mean()
        t_d = t_d / (mean_td + self.eps)

        d = pdist(student, squared=False)
        mean_d = d.mean()
        d = d / (mean_d + self.eps)

        loss = F.smooth_l1_loss(d, t_d, reduction="mean")
        return loss


class RkdLoss(nn.Layer):
    def __init__(self, eps=1e-12):
        super(RkdLoss, self).__init__()
        self.rkd_angle_loss_func = RKdAngle()
        self.rkd_angle_dist_func = RkdDistance()

    def forward(self, student, teacher):
        loss_dict = {}
        if isinstance(student, paddle.Tensor):
            loss_dict["rkd_angle_loss"] = self.rkd_angle_loss_func(student,
                                                                   teacher)
            loss_dict["rkd_dist_loss"] = self.rkd_angle_dist_func(student,
                                                                  teacher)
        else:
            s_names = list(student.keys())
            t_names = list(teacher.keys())
            for idx in range(len(s_names)):
                s_name = s_names[idx]
                t_name = t_names[idx]
                loss_dict["{}_rkd_angle_loss".format(
                    s_name)] = self.rkd_angle_loss_func(student[s_name],
                                                        teacher[t_name])
                loss_dict["{}_rkd_angle_loss".format(
                    s_name)] = self.rkd_angle_dist_func(student[s_name],
                                                        teacher[t_name])
        return loss_dict


if __name__ == "__main__":
    paddle.set_device("cpu")
    student = paddle.rand([32, 128, 10, 20])
    teacher = paddle.rand([32, 256, 10, 20])
    rkd_angle_func = RKdAngle()
    loss = rkd_angle_func(student, teacher)
    print(loss)

    rkd_dist_func = RkdDistance()
    loss = rkd_dist_func(student, teacher)
    print(loss)
