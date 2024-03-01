import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import tqdm
import time


def presample(nids, g, hop, batch_size=100000):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    layer_member = []
    layer_member.append(nids)
    for i in range(hop):
        neighbor = []
        for start in tqdm.trange(0, nids.numel(), batch_size):
            end = min(nids.numel(), start + batch_size)
            frontier, _, _ = sampler.sample_blocks(g, nids[start:end])
            neighbor.append(frontier)
        nids = torch.cat(neighbor).unique()
        layer_member.append(nids)
    return layer_member


class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            self.layers.append(dglnn.SAGEConv(in_dim, out_dim, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def layerwise_inference(self,
                            g,
                            batch_size,
                            nids=None,
                            fanout=None,
                            prob=None,
                            device="cuda",
                            sample_device="cuda"):

        if fanout is None:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        else:
            # only support equal fanout for each layer
            sampler = dgl.dataloading.NeighborSampler(fanout)

        if nids is None:
            nodes = torch.arange(g.num_nodes()).to(sample_device)
            dataloader = dgl.dataloading.DataLoader(
                g,
                nodes,
                sampler,
                device=sample_device,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0 if sample_device == "cuda" else 4,
                use_ddp=True,
                use_uva=True if sample_device == "cuda" else False)
        else:
            presample_tic = time.time()
            nodes_list = presample(nids,
                                   g,
                                   self.n_layers - 1,
                                   batch_size=batch_size)
            torch.cuda.synchronize()
            self.layer_presample_time += time.time() - presample_tic
            dataloader_list = []
            map_list = []

            for l in range(self.n_layers):
                dataloader_list.append(
                    dgl.dataloading.DataLoader(
                        g,
                        nodes_list[l],
                        sampler,
                        device=sample_device,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False,
                        num_workers=0 if sample_device == "cuda" else 4,
                        use_ddp=True,
                        use_uva=True if sample_device == "cuda" else False))
                map = torch.full((g.num_nodes(), ), -1)
                map[nodes_list[l]] = torch.arange(0, nodes_list[l].numel())
                map_list.append(map)

        feature = g.ndata["features"]
        for l, layer in enumerate(self.layers):
            if nids is None:
                y = torch.empty((
                    g.num_nodes(),
                    self.n_hidden if l != len(self.layers) -
                    1 else self.n_classes,
                ),
                                dtype=feature.dtype)
                if sample_device == "cuda":
                    torch.cuda.synchronize()
                    sampling_start = time.time()
                    for input_nodes, output_nodes, blocks in tqdm.tqdm(
                            dataloader):
                        blocks[0] = blocks[0].to(device)
                        torch.cuda.synchronize()
                        sampling_end = time.time()
                        self.layer_sample_times[
                            l] += sampling_end - sampling_start
                        loading_start = time.time()
                        x = feature[input_nodes.cpu()].to(device)
                        torch.cuda.synchronize()
                        loading_end = time.time()
                        self.layer_load_times[l] += loading_end - loading_start
                        forward_start = time.time()
                        h = layer(blocks[0], x)
                        if l != len(self.layers) - 1:
                            h = self.activation(h)
                            h = self.dropout(h)
                        torch.cuda.synchronize()
                        forward_end = time.time()
                        self.layer_forward_times[
                            l] += forward_end - forward_start
                        saving_start = time.time()
                        y[output_nodes] = h.to("cpu")
                        torch.cuda.synchronize()
                        saving_end = time.time()
                        self.layer_save_times[l] += saving_end - saving_start
                        sampling_start = time.time()
                else:
                    with dataloader.enable_cpu_affinity():
                        torch.cuda.synchronize()
                        sampling_start = time.time()
                        for input_nodes, output_nodes, blocks in tqdm.tqdm(
                                dataloader):
                            blocks[0] = blocks[0].to(device)
                            torch.cuda.synchronize()
                            sampling_end = time.time()
                            self.layer_sample_times[
                                l] += sampling_end - sampling_start
                            loading_start = time.time()
                            x = feature[input_nodes.cpu()].to(device)
                            torch.cuda.synchronize()
                            loading_end = time.time()
                            self.layer_load_times[
                                l] += loading_end - loading_start
                            forward_start = time.time()
                            h = layer(blocks[0], x)
                            if l != len(self.layers) - 1:
                                h = self.activation(h)
                                h = self.dropout(h)
                            torch.cuda.synchronize()
                            forward_end = time.time()
                            self.layer_forward_times[
                                l] += forward_end - forward_start
                            saving_start = time.time()
                            y[output_nodes] = h.to("cpu")
                            torch.cuda.synchronize()
                            saving_end = time.time()
                            self.layer_save_times[
                                l] += saving_end - saving_start
                            sampling_start = time.time()
                feature = y
            else:
                layer_nodes = nodes_list[self.n_layers - 1 - l]
                dataloader = dataloader_list[self.n_layers - 1 - l]
                map = map_list[self.n_layers - 1 - l]
                if l > 0:
                    last_map = map_list[self.n_layers - l]
                y = torch.empty((
                    layer_nodes.numel(),
                    self.n_hidden if l != len(self.layers) -
                    1 else self.n_classes,
                ),
                                dtype=feature.dtype)
                if sample_device == "cuda":
                    torch.cuda.synchronize()
                    sampling_start = time.time()
                    for input_nodes, output_nodes, blocks in tqdm.tqdm(
                            dataloader):
                        blocks[0] = blocks[0].to(device)
                        torch.cuda.synchronize()
                        sampling_end = time.time()
                        self.layer_sample_times[
                            l] += sampling_end - sampling_start
                        loading_start = time.time()
                        if l == 0:
                            x = feature[input_nodes.cpu()].to(device)
                        else:
                            x = feature[last_map[input_nodes.cpu()]].to(device)
                        torch.cuda.synchronize()
                        loading_end = time.time()
                        self.layer_load_times[l] += loading_end - loading_start
                        forward_start = time.time()
                        h = layer(blocks[0], x)
                        if l != len(self.layers) - 1:
                            h = self.activation(h)
                            h = self.dropout(h)
                        torch.cuda.synchronize()
                        forward_end = time.time()
                        self.layer_forward_times[
                            l] += forward_end - forward_start
                        saving_start = time.time()
                        y[map[output_nodes.cpu()]] = h.to("cpu")
                        torch.cuda.synchronize()
                        saving_end = time.time()
                        self.layer_save_times[l] += saving_end - saving_start
                        sampling_start = time.time()
                else:
                    with dataloader.enable_cpu_affinity():
                        torch.cuda.synchronize()
                        sampling_start = time.time()
                        for input_nodes, output_nodes, blocks in tqdm.tqdm(
                                dataloader):
                            blocks[0] = blocks[0].to(device)
                            torch.cuda.synchronize()
                            sampling_end = time.time()
                            self.layer_sample_times[
                                l] += sampling_end - sampling_start
                            loading_start = time.time()
                            if l == 0:
                                x = feature[input_nodes.cpu()].to(device)
                            else:
                                x = feature[last_map[input_nodes.cpu()]].to(
                                    device)
                            torch.cuda.synchronize()
                            loading_end = time.time()
                            self.layer_load_times[
                                l] += loading_end - loading_start
                            forward_start = time.time()
                            h = layer(blocks[0], x)
                            if l != len(self.layers) - 1:
                                h = self.activation(h)
                                h = self.dropout(h)
                            torch.cuda.synchronize()
                            forward_end = time.time()
                            self.layer_forward_times[
                                l] += forward_end - forward_start
                            saving_start = time.time()
                            y[map[output_nodes.cpu()]] = h.to("cpu")
                            torch.cuda.synchronize()
                            saving_end = time.time()
                            self.layer_save_times[
                                l] += saving_end - saving_start
                            sampling_start = time.time()
                feature = y

        return y

    def nodewise_inference(self,
                           g,
                           batch_size,
                           nids,
                           fanout=None,
                           prob=None,
                           device="cuda",
                           sample_device="cuda"):
        if fanout is None:
            if sample_device == "cuda":
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
                    # self.n_layers, prefetch_node_feats=["features"])
                    self.n_layers)
            else:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
                    self.n_layers)
        else:
            if sample_device == "cuda":
                sampler = dgl.dataloading.NeighborSampler(
                    # fanout, prefetch_node_feats=["features"])
                    fanout)
            else:
                sampler = dgl.dataloading.NeighborSampler(fanout)

        dataloader = dgl.dataloading.DataLoader(
            g,
            nids.to(sample_device),
            sampler,
            device=sample_device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0 if sample_device == "cuda" else 4,
            use_ddp=True,
            use_uva=True if sample_device == "cuda" else False)

        result = torch.full((nids.numel(), self.n_classes),
                            -1,
                            dtype=torch.float)
        map = torch.full((g.num_nodes(), ), -1)
        map[nids.cpu()] = torch.arange(0, nids.numel())

        if sample_device == "cuda":
            torch.cuda.synchronize()
            sampling_start = time.time()
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                blocks = [block.to(device) for block in blocks]
                torch.cuda.synchronize()
                sampling_end = time.time()
                self.node_sample_time += sampling_end - sampling_start
                loading_start = time.time()
                # batch_inputs = blocks[0].srcdata["features"].to(device)
                batch_inputs = g.ndata["features"][input_nodes.cpu()].to(
                    device)
                torch.cuda.synchronize()
                loading_end = time.time()
                self.node_load_time += loading_end - loading_start
                forward_start = time.time()
                pred = self.forward(blocks, batch_inputs)
                torch.cuda.synchronize()
                forward_end = time.time()
                self.node_forward_time += forward_end - forward_start
                save_start = time.time()
                result[map[output_nodes.cpu()]] = pred.cpu()
                torch.cuda.synchronize()
                save_end = time.time()
                self.node_save_time += save_end - save_start
                sampling_start = time.time()
        else:
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    blocks = [block.to(device) for block in blocks]
                    torch.cuda.synchronize()
                    sampling_end = time.time()
                    self.node_sample_time += sampling_end - sampling_start
                    loading_start = time.time()
                    batch_inputs = g.ndata["features"][input_nodes.cpu()].to(
                        device)
                    torch.cuda.synchronize()
                    loading_end = time.time()
                    self.node_load_time += loading_end - loading_start
                    forward_start = time.time()
                    pred = self.forward(blocks, batch_inputs)
                    torch.cuda.synchronize()
                    forward_end = time.time()
                    self.node_forward_time += forward_end - forward_start
                    save_start = time.time()
                    result[map[output_nodes.cpu()]] = pred.cpu()
                    torch.cuda.synchronize()
                    save_end = time.time()
                    self.node_save_time += save_end - save_start
                    sampling_start = time.time()

        return result

    def check_layerwise_recorder(self):
        timetable = ""
        timetable += "======Layerwise Time Recorder======\n"
        timetable += "Presampling time {:.3f}\n".format(
            self.layer_presample_time / self.epoch_num)
        for l in range(self.n_layers):
            timetable += "Layer {} sampling time {:.3f}\n".format(
                l, self.layer_sample_times[l] / self.epoch_num)
            timetable += "Layer {} loading time {:.3f}\n".format(
                l, self.layer_load_times[l] / self.epoch_num)
            timetable += "Layer {} forward time {:.3f}\n".format(
                l, self.layer_forward_times[l] / self.epoch_num)
            timetable += "Layer {} saving time {:.3f}\n".format(
                l, self.layer_save_times[l] / self.epoch_num)
        timetable += "===================================\n"
        print(timetable)

    def check_nodewise_recorder(self):
        timetable = ""
        timetable += "======Nodewise Time Recorder=======\n"
        timetable += "Sampling time {:.3f}\n".format(self.node_sample_time /
                                                     self.epoch_num)
        timetable += "Loading time {:.3f}\n".format(self.node_load_time /
                                                    self.epoch_num)
        timetable += "Forward time {:.3f}\n".format(self.node_forward_time /
                                                    self.epoch_num)
        timetable += "Saving time {:.3f}\n".format(self.node_save_time /
                                                   self.epoch_num)
        timetable += "===================================\n"
        print(timetable)

    def reset_recorder(self):
        self.epoch_num = 0
        self.node_sample_time = 0.0
        self.node_load_time = 0.0
        self.node_forward_time = 0.0
        self.node_save_time = 0.0
        self.layer_presample_time = 0.0
        self.layer_sample_times = []
        self.layer_load_times = []
        self.layer_forward_times = []
        self.layer_save_times = []
        for l in range(self.n_layers):
            self.layer_forward_times.append(0.0)
            self.layer_sample_times.append(0.0)
            self.layer_load_times.append(0.0)
            self.layer_save_times.append(0.0)

    def inc_epoch_count(self):
        self.epoch_num += 1


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
