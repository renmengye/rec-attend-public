import cv2
import numpy as np
import time
import os
from utils import logger
from data_api import orientation


def create_analyzer(name, display_name=None, fname=None):
    if display_name is None:
        display_name = name
    name = name.lower()
    if name == 'sbd':
        return StatsAnalyzer(display_name, f_symmetric_best_dice, fname=fname)
    if name == 'wt_cov':
        return StatsAnalyzer(display_name, f_wt_coverage, fname=fname)
    if name == 'unwt_cov':
        return StatsAnalyzer(display_name, f_unwt_coverage, fname=fname)
    if name == 'fg_dice':
        return StatsAnalyzer(display_name, f_fg_dice, fname=fname)
    if name == 'fg_iou':
        return StatsAnalyzer(display_name, f_fg_iou, fname=fname)
    if name == 'fg_iou_all':
        return ForegroundIOUAnalyzer(display_name, fname=fname)
    if name == 'bg_iou_all':
        return BackgroundIOUAnalyzer(display_name, fname=fname)
    if name == 'avg_fp':
        return StatsAnalyzer(display_name, f_fp, fname=fname)
    if name == 'avg_fn':
        return StatsAnalyzer(display_name, f_fn, fname=fname)
    if name == 'avg_pr':
        return StatsAnalyzer(display_name, f_pixel_pr, fname=fname)
    if name == 'avg_re':
        return StatsAnalyzer(display_name, f_pixel_re, fname=fname)
    if name == 'obj_pr':
        return StatsAnalyzer(display_name, f_obj_pr, fname=fname)
    if name == 'obj_re':
        return StatsAnalyzer(display_name, f_obj_re, fname=fname)
    if name == 'count_acc':
        return StatsAnalyzer(display_name, f_count_acc, fname=fname)
    if name == 'count_mse':
        return StatsAnalyzer(display_name, f_count_mse, fname=fname)
    if name == 'count':
        return CountAnalyzer(fname=fname)
    if name == 'dic':
        return StatsAnalyzer(display_name, f_dic, fname=fname)
    if name == 'dic_abs':
        return StatsAnalyzer(display_name, f_dic_abs, fname=fname)
    raise Exception('Analyzer not found: {}'.format(name))


class AnalyzerBase(object):

    def __init__(self, name):
        self.name = name
        self.log = logger.get()

    def stage(self, results):
        """Record one batch."""
        pass

    def finalize(self):
        """Finalize statistics."""
        pass


class CountAnalyzer(AnalyzerBase):

    def __init__(self, fname, name='count_rec'):
        self.fname = fname
        with open(fname, 'w') as f:
            f.write('Image ID,Count Out,Count GT\n')
        super(CountAnalyzer, self).__init__(name)
        pass

    def stage(self, results):
        s_out = results['s_out']
        s_gt = results['s_gt']
        y_out = results['y_out']
        count_out = f_count_out(y_out).sum(axis=1)
        count_gt = s_gt.sum(axis=1)
        indices = results['indices']
        with open(self.fname, 'a') as f:
            for ii, idx in enumerate(indices):
                f.write('{},{:d},{:d}\n'.format(
                    idx, int(count_out[ii]), int(count_gt[ii])))
                self.log.info('img id {} out {:d} gt {:d}'.format(
                    idx, int(count_out[ii]), int(count_gt[ii])))
                pass
            pass
        pass
    pass


class RenderInstanceAnalyzer(AnalyzerBase):

    def __init__(self, folder, dataset, semantic_labels=None):
        self.folder = folder
        self.dataset = dataset
        self.semantic_labels = semantic_labels
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.cmap = np.array([[192, 57, 43],
                              [243, 156, 18],
                              [26, 188, 156],
                              [41, 128, 185],
                              [142, 68, 173],
                              [44, 62, 80],
                              [127, 140, 141],
                              [17, 75, 95],
                              [2, 128, 144],
                              [228, 253, 225],
                              [69, 105, 144],
                              [244, 91, 105],
                              [91, 192, 235],
                              [253, 231, 76],
                              [155, 197, 61],
                              [229, 89, 52],
                              [250, 121, 33],
                              [124, 82, 47],
                              [86, 15, 94],
                              [38, 63, 77],
                              [1, 52, 55],
                              [319, 29, 82]], dtype='uint8')
        super(RenderInstanceAnalyzer, self).__init__('render_ins')
        if folder is not None:
            self.log.info('Writing output image to {}'.format(folder))
        else:
            self.log.fatal('No output folder')
        pass

    def stage(self, results):
        y_out = results['y_out']
        indices = results['indices']
        num_ex = len(y_out)
        for ii in range(num_ex):
            total_img = np.zeros(
                [y_out[ii].shape[1], y_out[ii].shape[2], 3], dtype='uint8')
            for jj in range(y_out[ii].shape[0]):
                y_out_ = (y_out[ii][jj]).astype('uint8')
                size = y_out_.sum()
                if size > 0:
                    total_img += np.expand_dims(y_out_, 2) * \
                        self.cmap[jj % self.cmap.shape[0]]
                pass
            # BGR => RGB
            total_img = total_img[:, :, [2, 1, 0]]
            output_fname = os.path.join(
                self.folder, self.dataset.get_fname(indices[ii]))
            cv2.imwrite(output_fname, total_img)
            pass
        pass
    pass


class RenderGroundtruthInstanceAnalyzer(RenderInstanceAnalyzer):

    def stage(self, results):
        y_out = results['y_out']
        y_gt = results['y_gt']
        iou_pairwise = results['iou_pairwise']
        indices = results['indices']
        num_ex = len(y_out)
        num_color = self.cmap.shape[0]
        for ii in range(num_ex):
            total_img = np.zeros(
                [y_gt[ii].shape[1], y_gt[ii].shape[2], 3], dtype='uint8')
            flag = np.zeros(num_color)
            for jj in range(y_gt[ii].shape[0]):
                y_gt_ = (y_gt[ii][jj]).astype('uint8')

                max_idx = np.argmax(iou_pairwise[ii][:, jj])
                if flag[max_idx] == 0:
                    color = self.cmap[max_idx]
                    flag[max_idx] = 1
                else:
                    for kk in range(num_color):
                        idx = num_color - kk - 1
                        if flag[idx] == 0:
                            color = self.cmap[idx]
                            flag[idx] = 1
                            break
                total_img += (total_img == 0).astype('uint8') * \
                    np.expand_dims(y_gt_, 2) * color
                pass
            # BGR => RGB
            total_img = total_img[:, :, [2, 1, 0]]
            output_fname = os.path.join(
                self.folder, self.dataset.get_fname(indices[ii]))
            cv2.imwrite(output_fname, total_img)
            pass
        pass
    pass


class RenderCityScapesOutputAnalyzer(AnalyzerBase):

    def __init__(self, folder, dataset):
        self.folder = folder
        self.dataset = dataset
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.labels = [('person', 24),
                       ('rider', 25),
                       ('car', 26),
                       ('truck', 27),
                       ('bus', 28),
                       ('train', 31),
                       ('motorcycle', 32),
                       ('bicycle', 33)]

        super(RenderCityScapesOutputAnalyzer, self).__init__('render_cs')
        if folder is not None:
            self.log.info('Writing output image to {}'.format(folder))
        else:
            self.log.fatal('No output folder')
        pass

    def stage(self, results):
        y_out = results['y_out']
        indices = results['indices']
        fg = results['y_in']
        score = results['conf']
        num_ex = len(y_out)
        for ii in range(num_ex):
            fn1 = self.dataset.get_fname(indices[ii])
            runname = fn1.split('_')[0]
            runfolder = os.path.join(self.folder, runname)
            if not os.path.exists(runfolder):
                os.makedirs(runfolder)
                pass
            text_fn = fn1.strip('.png') + '.txt'
            text_fn = os.path.join(runfolder, text_fn)
            with open(text_fn, 'w') as text_file:
                for jj in range(y_out[ii].shape[0]):
                    if score[ii][jj] > 0.5:
                        seg = y_out[ii][jj]
                        channel_vec = (np.expand_dims(seg, -1) * fg[ii]).mean(
                            axis=0).mean(axis=0)
                        # sem_idx = np.argmax(channel_vec) - 1
                        # if sem_idx != -1:
                        #     label_num = self.labels[sem_idx][1]
                        #     img_file = fn1.strip(
                        #         '.png') + '_{:03d}.png'.format(jj)
                        #     output_fname = os.path.join(runfolder, img_file)
                        #     cv2.imwrite(
                        #         output_fname, (seg * 255).astype('uint8'))
                        #     text_file.write('{} {:d} {:f}\n'.format(
                        #         img_file, label_num, score[ii, jj]))
                        # pass

                        if channel_vec[0] <= 0.7:
                            sem_idx = np.argmax(channel_vec[1:])
                            label_num = self.labels[sem_idx][1]
                            img_file = fn1.strip(
                                '.png') + '_{:03d}.png'.format(jj)
                            output_fname = os.path.join(runfolder, img_file)
                            cv2.imwrite(
                                output_fname, (seg * 255).astype('uint8'))
                            text_file.write('{} {:d} {:f}\n'.format(
                                img_file, label_num, score[ii, jj]))
                        pass
                    pass
                pass
            pass
        pass
    pass


class RenderOrientationAnalyzer(RenderInstanceAnalyzer):

    def stage(self, results):
        d_out = results['d_out']
        mask = results['mask']
        indices = results['indices']
        num_ex = len(mask)
        for ii in range(num_ex):
            img = orientation.build_orientation_img(d_out[ii], mask[ii])
            output_fname = os.path.join(
                self.folder, self.dataset.get_fname(indices[ii]))
            self.log.info(output_fname)
            cv2.imwrite(output_fname, img)
    pass


class RenderForegroundAnalyzer(AnalyzerBase):

    def __init__(self, folder, dataset):
        self.folder = folder
        self.dataset = dataset
        if not os.path.exists(folder):
            os.makedirs(folder)
        super(RenderForegroundAnalyzer, self).__init__('render_fg')
        if folder is not None:
            self.log.info('Writing output image to {}'.format(folder))
        else:
            self.log.fatal('No output folder')
        pass

    def stage(self, results):
        y_out = results['y_out']
        indices = results['indices']
        num_ex = len(y_out)
        for ii in range(num_ex):
            y_out_ii = (y_out[ii] * 255).astype('uint8')
            output_fname = os.path.join(
                self.folder, self.dataset.get_fname(indices[ii]))
            cv2.imwrite(output_fname, y_out_ii)
            pass
        pass
    pass


def f_iou(a, b):
    """IOU between two segmentations.

    Args:
        a: [..., H, W], binary mask
        b: [..., H, W], binary mask

    Returns:
        dice: [...]
    """
    inter = (a * b).sum(axis=-1).sum(axis=-1)
    union = (a + b).sum(axis=-1).sum(axis=-1) - inter
    return inter / (union + np.equal(union, 0).astype('float32'))


def f_iou_pairwise(a, b):
    """Pairwise IOU between two set of segmentations.
    """
    a = np.expand_dims(a, 1)
    b = np.expand_dims(b, 0)
    return f_iou(a, b)


def _f_pr(a, b):
    """Precision between two segmentations, denominator is the 1st argument.

    Args:
        a: [..., H, W], binary mask
        b: [..., H, W], binary mask

    Returns:
        dice: [...]
    """
    inter = (a * b).sum(axis=-1).sum(axis=-1)
    asum = a.sum(axis=-1).sum(axis=-1)
    return inter / (asum + np.equal(asum, 0).astype('float32'))


def _f_dice(a, b):
    """DICE between two segmentations.

    Args:
        a: [..., H, W], binary mask
        b: [..., H, W], binary mask

    Returns:
        dice: [...]
    """
    card_a = a.sum(axis=-1).sum(axis=-1)
    card_b = b.sum(axis=-1).sum(axis=-1)
    card_ab = (a * b).sum(axis=-1).sum(axis=-1)
    card_sum = card_a + card_b
    dice = 2 * card_ab / (card_sum + np.equal(card_sum, 0).astype('float32'))
    return dice


def _f_best_dice(a, b):
    """For each a, look for the best DICE of all b.

    Args:
        a: [T, H, W], binary mask
        b: [T, H, W], binary mask

    Returns:
        best_dice: [T]
    """
    bd = np.zeros([a.shape[0]])
    for ii in range(a.shape[0]):
        a_ = a[ii: ii + 1, :, :]
        dice = _f_dice(a_, b)
        bd[ii] = dice.max(axis=0)
        pass
    return bd


def _f_match(iou_pairwise):
    """Get maximum weighted bipartite matching of a square matrix.

    Args:
        iou_pairwise: [N, N], weights of the bipartite graph to maximize.

    Returns:
        match: [N, N], binary mask
    """
    sess = tf.Session()
    tf_match = tf.user_ops.hungarian(
        tf.constant(iou_pairwise.astype('float32')))[0]
    return tf_match.eval(session=sess)


def f_ins_iou(results):
    """Calculates average instance-level IOU..

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        ins_iou: [B]
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_out = results['s_out']
    s_gt = results['s_gt']
    num_obj = _f_num_obj(s_gt)
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    ins_iou = np.zeros([num_ex])
    for ii in range(num_ex):
        iou_pairwise_ = results['iou_pairwise'][ii]
        iou_pairwise_ = np.maximum(1e-4, iou_pairwise)
        iou_pairwise_ = np.round(iou_pairwise * 1e4) / 1e4
        match = _f_match(iou_pairwise)
        match[num_obj[ii]:, :] = 0.0
        match[:, num_obj[ii]:] = 0.0
        ins_iou[ii] = (iou_pairwise * match).sum(
            axis=-1).sum(axis=-1) / num_obj[ii]
    return ins_iou


def f_symmetric_best_dice(results):
    """Calculates symmetric best DICE. min(BestDICE(a, b), BestDICE(b, a)).

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        sbd: [B]
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_out = results['s_out']
    s_gt = results['s_gt']
    num_obj = _f_num_obj(s_gt)

    def f_bd(a, b):
        num_ex = len(a)
        timespan = a[0].shape[0]
        bd = np.zeros([num_ex, timespan])
        for ii in range(num_ex):
            bd[ii] = _f_best_dice(a[ii], b[ii])
        bd_mean = np.zeros([num_ex])
        for ii in range(num_ex):
            bd_mean[ii] = bd[ii, :int(num_obj[ii])].mean()
        return bd_mean
    return np.minimum(f_bd(y_out, y_gt), f_bd(y_gt, y_out))


def f_coverage_single(iou_pairwise):
    return iou_pairwise.max(axis=0)


def f_coverage_weights(y_gt, num_obj, weighted=False):
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    weights = np.zeros([num_ex, timespan])
    for ii in range(num_ex):
        if weighted:
            y_gt_sum = y_gt[ii].sum()
            weights[ii] = y_gt[ii].sum(axis=-1).sum(axis=-1) / \
                (y_gt_sum + np.equal(y_gt_sum, 0).astype('float32'))
        else:
            weights[ii] = 1 / num_obj[ii]
    return weights


def f_coverage(results, weighted=False):
    """Calculates coverage score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        cov: [B]
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_gt = results['s_gt']
    num_obj = _f_num_obj(s_gt)
    num_ex = len(y_gt)
    cov = np.array([f_coverage_single(iou_)
                    for iou_ in results['iou_pairwise']])
    weights = f_coverage_weights(y_gt, num_obj, weighted=weighted)
    cov *= weights
    cov_mean = np.zeros([num_ex])
    for ii in range(num_ex):
        cov_mean[ii] = cov[ii, :int(num_obj[ii])].sum()
        pass
    return cov_mean


def f_wt_coverage(results):
    """Calculates weighted coverage score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        cov: [B]
    """
    return f_coverage(results, weighted=True)


def f_unwt_coverage(results):
    """Calculates unweighted coverage score.

    Args:
        a: list of [T, H, W], binary mask
        b: list of [T, H, W], binary mask

    Returns:
        cov: [B]
    """
    return f_coverage(results, weighted=False)


def f_fg_iou(results):
    """Calculates foreground IOU score.

    Args:
        a: list of [T, H, W] or [H, W], binary mask
        b: list of [T, H, W] or [H, W], binary mask

    Returns:
        fg_iou: [B]
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    num_ex = len(y_gt)
    fg_iou = np.zeros([num_ex])
    if len(y_gt[0].shape) == 3:
        for ii in range(num_ex):
            fg_iou[ii] = f_iou(y_out[ii].max(axis=0), y_gt[ii].max(axis=0))
    else:
        for ii in range(num_ex):
            fg_iou[ii] = f_iou(y_out[ii], y_gt[ii])
    return fg_iou


def f_fg_dice(results):
    """Calculates foreground DICE score.

    Args:
        a: list of [T, H, W] or [H, W], binary mask
        b: list of [T, H, W] or [H, W], binary mask

    Returns:
        fg_dice: [B]
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    num_ex = len(y_gt)
    fg_dice = np.zeros([num_ex])
    if len(y_gt[0].shape) == 3:
        for ii in range(num_ex):
            fg_dice[ii] = _f_dice(y_out[ii].max(axis=0), y_gt[ii].max(axis=0))
    else:
        for ii in range(num_ex):
            fg_dice[ii] = _f_dice(y_out[ii], y_gt[ii])
    return fg_dice


def f_fp(results):
    """Calculates number of false positive instances."""
    y_out = results['y_out']
    s_out = results['s_out']
    s_gt = results['s_gt']
    num_ex = s_out.shape[0]
    fp = np.zeros([num_ex])
    for ii in range(num_ex):
        y_out_sum = (y_out[ii].sum(axis=-1).sum(axis=-1) > 0).astype('float32')
        iou_pairwise_ = results['iou_pairwise'][ii]
        unmatch_sum = np.equal(iou_pairwise_.sum(axis=1), 0).astype('float32')
        fp[ii] = (y_out_sum * unmatch_sum).sum()
        pass
    return fp


def f_fn(results):
    """Calculates number of false negative instances."""
    s_gt = results['s_gt']
    num_ex = s_gt.shape[0]
    fn = np.zeros([num_ex])
    for ii in range(num_ex):
        iou_pairwise_ = results['iou_pairwise'][ii]
        unmatch_sum = np.equal(iou_pairwise_.sum(axis=0), 0).astype('float32')
        fn[ii] = (s_gt[ii] * unmatch_sum).sum()
        pass
    return fn


def f_pixel_pr(results):
    """Calculates pixel-level instance precision.
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_gt = results['s_gt']
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    pix_pr = []
    num_obj = _f_num_obj(s_gt)
    count_out = f_count_out(y_out)
    for ii in range(num_ex):
        y_out_ = y_out[ii]
        y_gt_sum = y_gt[ii].max(axis=0, keepdims=True)
        pr_ = _f_pr(y_out_, y_gt_sum)
        for jj in range(timespan):
            if count_out[ii, jj] > 0:
                pix_pr.append(pr_[jj])
    pix_pr = np.array(pix_pr)
    return pix_pr


def f_pixel_re(results):
    """Calculates pixel-level instance recall.

    For each groundtruth instance, calculate the percentage of the pixels in 
    any of the prediction foreground. Average all recall rate.
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_gt = results['s_gt']
    num_ex = len(y_gt)
    pix_re = []
    count_gt = s_gt.sum(axis=1)
    for ii in range(num_ex):
        y_gt_ = y_gt[ii]
        y_out_sum = y_out[ii].max(axis=0, keepdims=True)
        re_ = _f_pr(y_gt_, y_out_sum)
        for jj in range(int(count_gt[ii])):
            pix_re.append(re_[jj])
        pass
    pix_re = np.array(pix_re)
    return pix_re


def f_obj_pr(results):
    """Calculates object-level precision."""
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_gt = results['s_gt']
    num_ex = len(y_gt)
    timespan = y_gt[0].shape[0]
    obj_pr = []
    num_obj = _f_num_obj(s_gt)
    count_out = f_count_out(y_out)
    for ii in range(num_ex):
        iou_pairwise_ = results['iou_pairwise'][ii]
        matched = (iou_pairwise_.max(axis=1) >= 0.5).astype('float32')
        for jj in range(timespan):
            if count_out[ii, jj] > 0:
                obj_pr.append(matched[jj])
        pass
    obj_pr = np.array(obj_pr)
    return obj_pr


def f_obj_re(results):
    """Calculates object-level recall."""
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_gt = results['s_gt']
    num_ex = len(y_gt)
    obj_re = []
    count_gt = s_gt.sum(axis=1)
    for ii in range(num_ex):
        iou_pairwise_ = results['iou_pairwise'][ii]
        matched = (iou_pairwise_.max(axis=0) >= 0.5).astype('float32')
        for jj in range(int(count_gt[ii])):
            obj_re.append(matched[jj])
        pass
    obj_re = np.array(obj_re)
    # print 'obj re', obj_re
    return obj_re


def f_count_mse(results):
    """Calculates count MSE.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        count_acc: [B]
    """
    s_out = results['s_out']
    s_gt = results['s_gt']
    y_out = results['y_out']
    count_out = f_count_out(y_out).sum(axis=1)
    count_gt = s_gt.sum(axis=1)
    return (count_out - count_gt).astype('float') ** 2


def f_count_acc(results):
    """Calculates count accuracy.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        count_acc: [B]
    """
    s_out = results['s_out']
    s_gt = results['s_gt']
    y_out = results['y_out']
    count_out = f_count_out(y_out).sum(axis=1)
    count_gt = s_gt.sum(axis=1)
    return (count_out == count_gt).astype('float')


def f_dic(results):
    """Calculates difference in count.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        dic: [B]
    """
    s_out = results['s_out']
    s_gt = results['s_gt']
    y_out = results['y_out']
    count_out = f_count_out(y_out).sum(axis=1)
    count_gt = s_gt.sum(axis=1)
    return (count_out - count_gt)


def f_dic_abs(results):
    """Calculates absolute difference in count.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        dic_abs: [B]
    """
    y_out = results['y_out']
    y_gt = results['y_gt']
    s_out = results['s_out']
    s_gt = results['s_gt']
    count_out = f_count_out(y_out).sum(axis=1)
    count_gt = s_gt.sum(axis=1)
    return np.abs(count_out - count_gt)


def f_count_out(y_out):
    count_out = [(_y.sum(axis=-1).sum(axis=-1) > 0).astype('float32')
                 for _y in y_out]
    count_out = np.array(count_out)
    return count_out


def _f_num_obj(s_gt):
    """Convert to count.

    Args:
        s_out: [B, T], binary mask
        s_gt: [B, T], binary mask

    Returns:
        count_out: [B]
        count_gt: [B]
        num_obj: [B]
    """
    count_gt = s_gt.sum(axis=1)
    num_obj = np.maximum(count_gt, 1)
    return num_obj


class StatsAnalyzer(AnalyzerBase):
    """Record average statistics."""

    def __init__(self, name, func, fname=None):
        self.sum = 0.0
        self.sum2 = 0.0
        self.num_ex = 0
        self.func = func
        self.fname = fname
        with open(self.fname, 'w') as f:
            f.write('ID,Score\n')
        super(StatsAnalyzer, self).__init__(name)
        pass

    def stage(self, results):
        """Record one batch."""
        start = time.time()
        _tmp = self.func(results)
        _num = _tmp.shape[0]
        self.num_ex += _num
        self.sum += _tmp.sum()
        self.sum2 += (_tmp ** 2).sum()
        end = time.time()
        elapsed_time = (end - start) * 1000
        with open(self.fname, 'a') as f:
            for ii in range(_num):
                f.write('{},{:.4f}\n'.format(0, _tmp[ii]))
        with self.log.verbose_level(2):
            self.log.info('{} finished in {:.2f}ms'.format(
                self.name, elapsed_time))
        pass

    def finalize(self):
        """Finalize statistics."""
        mean = self.sum / self.num_ex
        std = np.sqrt(np.maximum(0.0, self.sum2 / self.num_ex - mean ** 2))
        self.log.info('{:17s}{:7.4f} ({:6.4f})'.format(self.name, mean, std))
        if self.fname is not None:
            with open(self.fname, 'a') as f:
                f.write('Mean,{:.4f}\n'.format(mean))
                f.write('Std,{:.4f}\n'.format(std))
        pass


class ForegroundIOUAnalyzer(AnalyzerBase):
    """Analyze IOU for an entire dataset (not per image)"""

    def __init__(self, name='FG IOU ALL', fname=None):
        self.inter = 0.0
        self.union = 0.0
        self.fname = fname
        super(ForegroundIOUAnalyzer, self).__init__(name)
        pass

    def stage(self, results):
        start = time.time()
        y_out = results['y_out']
        y_gt = results['y_gt']
        for a, b in zip(y_out, y_gt):
            if len(a.shape) == 3:
                a = a.max(axis=0)
                b = b.max(axis=0)
            _inter = (a * b).sum()
            _union = a.sum() + b.sum() - _inter
            self.inter += _inter
            self.union += _union
            pass
        end = time.time()
        elapsed_time = (end - start) * 1000
        with self.log.verbose_level(2):
            self.log.info('{} finished in {:.2f}ms'.format(
                self.name, elapsed_time))
        pass

    def finalize(self):
        iou = self.inter / self.union
        self.log.info('{:17s}{:7.4f}'.format(self.name, iou))
        pass


class BackgroundIOUAnalyzer(AnalyzerBase):
    """Analyze IOU for an entire dataset (not per image)"""

    def __init__(self, name='BG IOU ALL', fname=None):
        self.inter = 0.0
        self.union = 0.0
        self.name = name
        self.fname = fname
        super(BackgroundIOUAnalyzer, self).__init__(name)
        pass

    def stage(self, results):
        start = time.time()
        y_out = results['y_out']
        y_gt = results['y_gt']
        for a, b in zip(y_out, y_gt):
            if len(a.shape) == 3:
                a = a.max(axis=0)
                b = b.max(axis=0)
            _a = 1 - a
            _b = 1 - b
            _inter = (_a * _b).sum()
            _union = _a.sum() + _b.sum() - _inter
            self.inter += _inter
            self.union += _union
            pass
        end = time.time()
        elapsed_time = (end - start) * 1000
        with self.log.verbose_level(2):
            self.log.info('{} finished in {:.2f}ms'.format(
                self.name, elapsed_time))
        pass

    def finalize(self):
        iou = self.inter / self.union
        self.log.info('{:17s}{:7.4f}'.format(self.name, iou))
        pass
