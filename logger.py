# -*- coding: utf-8 -*-
import sys
import os


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        #        self.log = open(filename, 'a')
        self.filename = filename

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        with open(self.filename, 'a') as log:
            log.write(message)

    def flush(self):
        pass


def build_holder(args):
    holder = ""
    if os.path.exists("/data/lgq/"):
        holder = "/data/lgq/"

    if args.seed >= 0:
        # holder = holder + "result/" + "seed=" + str(args.seed) + "/"
        holder = holder + "result_new/" + "seed=" + str(args.seed) + "/"
    else:
        holder = holder + "result/" + "seed=random" + "/"

    holder = holder + args.agent + "_" + args.data + "_" + args.classify + "_" + str(args.retrieve)
    if args.retrieve == "pearson":
        holder = holder + "=" + str(args.subsample)

    holder = holder + "_" + args.loss

    if args.dist:
        holder = holder + "_dis_" + "_τ=" + str(args.temperature)

    if args.loss == "focal":
        holder = holder + "_α=" + str(args.focal_alpha) + "_γ=" + str(args.focal_gamma)
    elif args.loss == "rfocal":
        holder = holder + "_α=" + str(args.rfl_alpha) + "_σ=" + str(args.rfl_sigma) + "_μ=" + str(args.rfl_miu)

    if args.kd_trick:
        holder = holder + "_" + str(args.kd_type)
        holder = holder + "_λ=" + str(args.kd_lamda)

        if args.kd_type != 'fkd' and args.kd_type != 'dist':
            holder = holder + "_T=" + str(args.T)

        if args.old_cls:
            holder = holder + "_old_cls"

    if args.pfkd:
        holder = holder + "_pfkd"
        holder = holder + "_λ=" + str(args.kd_lamda)

    if args.agent == "scr":
        holder = holder + "_τ=" + str(args.temperature)

    if args.bf:
        holder = holder + "_BF"
        holder = holder + "_drop=" + str(args.drop_rate)

    if args.data_aug:
        holder = holder + "_aug"

    holder = holder + "_eps=" + str(args.eps_mem_batch)
    holder = holder + "_mem=" + str(args.mem_size)
    holder = holder + "_lr=" + str(args.learning_rate)

    if args.review_trick:
        holder = holder + "_rev"

    if args.fix_order:
        holder = holder + "_fix"

    if args.certain_filter:
        holder = holder + "_filter=" + str(args.filter_keep)

    if not os.path.exists(holder):
        os.makedirs(holder)

    return holder
