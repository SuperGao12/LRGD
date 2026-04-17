
def print_intra_acc(RGB_true_label, IR_true_label, labels_RGB, labels_IR):

    N_rgb = labels_RGB.shape[0]
    N_ir = labels_IR.shape[0]

    RGB_mask_in = RGB_true_label.expand(N_rgb, N_rgb).eq(RGB_true_label.expand(N_rgb, N_rgb).t())
    IR_mask_in = IR_true_label.expand(N_ir, N_ir).eq(IR_true_label.expand(N_ir, N_ir).t())

    RGB_mask_in_p = labels_RGB.expand(N_rgb, N_rgb).eq(labels_RGB.expand(N_rgb, N_rgb).t())
    IR_mask_in_p = labels_IR.expand(N_ir, N_ir).eq(labels_IR.expand(N_ir, N_ir).t())

    RGB_in_acc = (RGB_mask_in_p * RGB_mask_in).sum() / RGB_mask_in.sum()
    IR_in_acc = (IR_mask_in_p * IR_mask_in).sum() / IR_mask_in.sum()
    RGB_in_recall = (RGB_mask_in_p * RGB_mask_in).sum() / RGB_mask_in_p.sum()
    IR_in_recall = (IR_mask_in_p * IR_mask_in).sum() / IR_mask_in_p.sum()

    print('RGB_in_recall:{:.4f} // IR_in_recall:{:.4f} // RGB_in_acc:{:.4f} // IR_in_acc:{:.4f}'
          .format(RGB_in_acc, IR_in_acc, RGB_in_recall, IR_in_recall))


def print_cm_acc(RGB_true_label, IR_true_label, labels_RGB, labels_IR, RGB_instance_IR_label, IR_instance_RGB_label):
    N_rgb = labels_RGB.shape[0]
    N_ir = labels_IR.shape[0]

    RGB_mask_c_p = labels_RGB.expand(N_ir, N_rgb).t().eq(IR_instance_RGB_label.expand(N_rgb, N_ir))
    IR_mask_c_p = RGB_instance_IR_label.expand(N_ir, N_rgb).t().eq(labels_IR.expand(N_rgb, N_ir))
    true_mask_c = RGB_true_label.expand(N_ir, N_rgb).t().eq(IR_true_label.expand(N_rgb, N_ir))

    RGB_acc = (RGB_mask_c_p * true_mask_c).sum() / true_mask_c.sum()
    IR_acc = (IR_mask_c_p * true_mask_c).sum() / true_mask_c.sum()
    RGB_recall = (RGB_mask_c_p * true_mask_c).sum() / RGB_mask_c_p.sum()
    IR_recall = (IR_mask_c_p * true_mask_c).sum() / IR_mask_c_p.sum()

    print('RGB_recall:{:.4f} // IR_recall:{:.4f} // RGB_acc:{:.4f} // IR_acc:{:.4f}'
          .format(RGB_acc, IR_acc, RGB_recall, IR_recall))


def conversion_(pseudo_labels, result):
    converted_labels = []
    for label in pseudo_labels:
        if label in result:
            converted_labels.append(result[label])
    return converted_labels