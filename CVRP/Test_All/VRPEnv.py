from dataclasses import dataclass
import torch

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

@dataclass
class Reset_State:
    problems: torch.Tensor


@dataclass
class Step_State:

    problems: torch.Tensor


class VRPEnv:
    def __init__(self, **env_params):

        self.env_params = env_params
        self.problem_size = None

        self.data_path = env_params['data_path']
        self.sub_path = env_params['sub_path']

        self.batch_size = None
        self.problems = None  # shape: [B,V,2]
        self.first_node = None  # shape: [B,V]

        self.loc_all=[] #  shape (B,V+1,2)
        self.demand_all=[] # shape (B,V+1)
        self.capacity_all=[] # shape (B)
        self.cost_all=[] # shape (B)
        self.solution_all=[] # shape (B,V+1,2)
        self.duration_all =[]
        self.start_capacity=None


        self.selected_count = None
        self.current_node = None

        self.selected_node_list = None
        self.selected_student_list = None

        self.decode_method = env_params['decode_method']

        self.test_in_vrplib = env_params['test_in_vrplib']
        self.vrplib_path = env_params['vrplib_path']
        self.vrplib_cost = None
        self.vrplib_name = None
        self.vrplib_problems = None
        self.problem_max_min = None
        self.episode = None

    def load_problems(self,episode, batch_size,problem_size_type=100, dataset_size = None,
                      current_best_solution=None, only_test = False, fix_length = None,
                      mode = 'repair',sub_path=False):
        self.episode = episode
        self.batch_size = batch_size

        if not self.test_in_vrplib:
            if only_test:

                self.problems_nodes = self.raw_data_nodes[episode:episode + batch_size]
                # shape (B,V+1,2)
                self.Batch_demand = self.raw_data_demand[episode:episode + batch_size]
                # shape (B,V+1)

                self.Batch_capacity = self.raw_data_capacity[episode:episode + batch_size]
                # shape (B)
                self.solution = self.raw_data_node_flag[episode:episode + batch_size]
                # shape (B,V,2)
                self.Batch_capacity = self.Batch_capacity[:,None].repeat(1,self.solution.shape[1]+1)
                # shape (B,V+1)

                self.problems = torch.cat((self.problems_nodes,self.Batch_demand[:,:,None],
                                           self.Batch_capacity[:,:,None]),dim=2)
                # shape (B,V+1,4)
            else:
                dataset = self.datas[str(problem_size_type) + '_' + str(dataset_size)]

                self.problems_nodes =  dataset['coor'][episode:episode + batch_size]
                # shape (B,V+1,2)
                self.Batch_demand =  dataset['demand'][episode:episode + batch_size]
                # shape (B,V+1)
                self.raw_data_capacity =dataset['capacity']

                self.Batch_capacity = self.raw_data_capacity[episode:episode + batch_size][:,None,None].\
                    repeat(1, problem_size_type+1, 1)


                self.problems = torch.cat((self.problems_nodes,self.Batch_demand[:,:,None],
                                           self.Batch_capacity),dim=2)
                self.solution = current_best_solution


            if sub_path:

                self.problems, self.solution = self.sampling_subpaths(self.problems, self.solution,length_fix=fix_length)

        else:

            problem_nodes = self.cvrp_node_coords[episode]
            problem_demands = self.cvrp_demands[episode]
            problem_capacitys = self.cvrp_capacitys[episode]

            problem_size = len(problem_nodes)

            problem_nodes = np.array(problem_nodes).reshape(1,problem_size,2)

            problem_demands = np.array(problem_demands).reshape(1, problem_size, 1)

            problem_nodes = torch.from_numpy(problem_nodes).cuda().float()

            self.problem_max_min = [torch.max(problem_nodes),torch.min(problem_nodes)]

            problem_nodes = (problem_nodes - self.problem_max_min[1])/(self.problem_max_min[0]-self.problem_max_min[1])

            problem_demands = torch.from_numpy(problem_demands).cuda().float()

            capacity_repeat = torch.tensor([problem_capacitys]).cuda().float().unsqueeze(0).unsqueeze(1).repeat(1,problem_size,1)
            self.raw_data_capacity = capacity_repeat
            self.problems = torch.cat((problem_nodes,problem_demands,capacity_repeat),dim=2)

            self.solution = None

        self.problem_size = self.problems.shape[1]-1





    def vrp_whole_and_solution_subrandom_inverse(self, solution):

        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])

            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        clockwise_or_not = torch.rand(len(sub_tours_padding))

        clockwise_or_not_bool = clockwise_or_not.le(0.5)


        sub_tours_padding[clockwise_or_not_bool] = torch.flip(sub_tours_padding[clockwise_or_not_bool], dims=[1])

        sub_tourss_back = sub_tourss

        sub_tourss_back[x5] = sub_tours_padding.ravel()

        solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(batch_size, problem_size)

        solution_flip = torch.cat((solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2)

        return solution_flip

    def vrp_whole_and_solution_subrandom_shift_V2inverse(self, solution):

        problem_size = solution.shape[1]
        batch_size = solution.shape[0]

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1
        end_with_depot[:,1] = torch.roll(end_with_depot[:,1],dims=0,shifts=-1)
        visit_depot_num = solution[:,:,1].sum(1)
        min_length = torch.min(visit_depot_num)

        first_node_index = torch.randint(low=0, high=min_length, size=[1])[0]  # in [0,N)

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        pick_end_with_depot_index = temp_index_torch + first_node_index
        pick_end_with_depot_ = end_with_depot[pick_end_with_depot_index][:,1]

        first_index= pick_end_with_depot_
        end_indeex = pick_end_with_depot_+problem_size

        index = torch.arange(2*problem_size)[None,:].repeat(batch_size,1)
        x1 = index > first_index[:,None]
        x2 = index<= end_indeex[:,None]
        x3 = x1.int()*x2.int()
        double_solution = solution.repeat(1,2,1)
        solution = double_solution[x3.gt(0.5)[:,:,None].repeat(1,1,2)].reshape(batch_size,problem_size,2)

        batch_size = solution.shape[0]
        problem_size = solution.shape[1]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).int()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        # 4.
        clockwise_or_not = torch.rand(len(sub_tours_padding))

        clockwise_or_not_bool = clockwise_or_not.le(0.5)

        # 5.
        sub_tours_padding[clockwise_or_not_bool] = torch.flip(sub_tours_padding[clockwise_or_not_bool], dims=[1])

        # 6。
        sub_tourss_back = sub_tourss

        sub_tourss_back[x5] = sub_tours_padding.ravel()

        solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(batch_size, problem_size)

        solution_flip = torch.cat((solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2)

        return solution_flip


    def sampling_subpaths(self, problems, solution, length_fix=False):

        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])

            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        problems_size = problems.shape[1] - 1

        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        if length_fix is None:
            length_of_subpath = problems_size
        else:
            length_of_subpath = length_fix

        clockwise_or_not = torch.rand(1)[0]

        if clockwise_or_not >= 0.5:
            solution = torch.flip(solution, dims=[1])

            index = torch.arange(solution.shape[1]).roll(shifts=1)
            solution[:, :, 1] = solution[:, index, 1]

        # 1.3

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        # 1.4
        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(select_end_with_depot_node_index).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        select_end_with_depot_node_index_ = select_end_with_depot_node_index + temp_index_torch

        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index_, 1]

        # 1.5
        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]

        sub_tour = double_solution[:, indexxxx, :]

        sub_tour = sub_tour.view(-1, length_of_subpath, 2)

        index_1 = torch.arange(0, batch_size * batch_size, batch_size)
        index_2 = torch.arange(batch_size)
        index_3 = index_1 + index_2
        sub_solution = sub_tour[index_3, :, :]

        offset_index = problems.shape[0]
        start_index = indexxxx[:,0]


        x1 = torch.arange(double_solution[:offset_index,:,1].shape[1])<=start_index[:offset_index][:,None]

        start_capacity = 0
        before_is_via_depot_all = double_solution[:offset_index,:,1]*x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2-1

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = select_end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[select_end_with_depot_node_index_2]

        before_start_index = before_is_via_depot_index[:,1]
        x2 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) <start_index[:offset_index][:, None]
        x3 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) >=before_start_index[:, None]

        x4 = x2 * x3

        double_solution_demand = problems[:offset_index,:,2][torch.arange(offset_index)[:,None].repeat(1,double_solution.shape[1]),double_solution[:offset_index,:,0] ]

        before_demand = double_solution_demand*x4

        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index,:,3] = problems[:offset_index,:,3] - self.satisfy_demand[:,None]

        sub_solution_node = sub_solution[:, :, 0]

        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
        sub_solution[:, :, 0] = new_sulution_rank+1


        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending, new_sulution_ascending, new_sulution_ascending), dim=1). \
            type(torch.long).sort(dim=-1, descending=False)

        # 构造torch高级索引
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)

        return new_data, sub_solution


    def sampling_subpaths_repair_V2(self, problems, solution,length_sub,first_index_sub,
                                    length_fix=False, mode='test', repair=True):

        solution = self.vrp_whole_and_solution_subrandom_shift_V2inverse(solution)

        problems_size = problems.shape[1] - 1

        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        length_of_subpath = length_sub  # in [4,N]

        factor_max = torch.sum(solution[:, :, 1],dim=1).min()
        # to aviod out of memory
        if factor_max>250:
            factor_max=250

        factor = int(problems_size/length_of_subpath)

        if factor>factor_max:
            factor=factor_max

        if not self.env_params['PRC']:
            factor=1

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot.clone()
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1
        end_with_depot[:,1] = torch.roll(end_with_depot[:,1],dims=0,shifts=-1)

        before_start_indexes = torch.zeros(batch_size, factor, 2, dtype=torch.long)
        start_indexes = torch.zeros(batch_size,factor,2,dtype=torch.long)
        end_indexes = torch.zeros(batch_size,factor,2,dtype=torch.long)

        def division_(start_from_depot):

            start_temp1 = start_from_depot[:, 0]
            start_temp2 = torch.roll(start_temp1, shifts=1, dims=0)
            start_temp3 = start_temp2 - start_temp1

            start_index_1 = start_temp3.nonzero()

            return start_index_1

        def fill_index(before_start_indexes,start_indexes, end_indexes,end_with_depot,end_index_1,length_sub,fff=0):

            start_index_1_0 = end_with_depot[end_index_1]

            end_indexes[:,[fff],:] = start_index_1_0
            if fff>0:
                before_start_indexes[:,[fff-1],:] = start_index_1_0
            start_index_1_0_temp =  start_index_1_0.clone()
            start_index_1_0_temp[:,:,1] -=length_sub
            start_indexes[:, [fff], :] = start_index_1_0_temp

            return before_start_indexes,start_indexes, end_indexes

        start_index_1 = division_(start_from_depot)
        if start_index_1.shape[0]==0:
            start_index_1=torch.tensor([0])

        all_depot_num = start_from_depot.shape[0]
        end_index_1 = torch.cat((start_index_1[1:].reshape(batch_size-1,1),torch.tensor([[all_depot_num]])),dim=0)-1

        division = end_index_1
        before_start_indexes,start_indexes, end_indexes =\
            fill_index(before_start_indexes,start_indexes, end_indexes,end_with_depot,end_index_1,length_sub,fff=0)

        def create_mask(all_depot_num,batch_size,division):
            index1 = torch.arange(all_depot_num)[None, :].repeat(batch_size, 1)
            temp1 = torch.cat((division.ravel(), torch.tensor([all_depot_num])), dim=0)

            temp2 = temp1[:batch_size]

            temp3 = temp1[1:batch_size + 1]

            temp2 = index1 >= temp2[:, None]

            temp3 = index1 < temp3[:, None]
            temp2 = temp2.int()
            temp3 = temp3.int()

            temp6 = temp2 * temp3

            return temp6, index1

        def select_next_depot_node( end_with_depot, start_indexes, fff,temp6,index1):

            if_start = end_with_depot[:, [1]] < start_indexes[:, fff - 1, 1]
            if_start = if_start.transpose(0, 1)

            temp7 =if_start * temp6

            temp8 = torch.roll(temp7, shifts=-1, dims=1)

            temp9 = temp7 - temp8

            select_index_next = index1[temp9.gt(0.5)]

            return select_index_next[:, None]

        def select_before_start_depot_node(start_from_depot, start_indexes, division,temp6,index1):

            if_start = start_from_depot[:, 1] <= start_indexes[:, 1][:, None]

            temp2 = if_start * temp6

            temp3 = torch.roll(temp2, shifts=-1, dims=1)

            temp2 = temp2 - temp3

            select_index_next = index1[temp2.gt(0.5)]

            return select_index_next[:, None]

        temp6,index1 = create_mask(all_depot_num,batch_size,start_index_1)

        for fff in range(1,factor):

            select_index_next = select_next_depot_node(end_with_depot,start_indexes,fff,temp6,index1)

            if select_index_next.ravel().shape[0]!=batch_size:
                factor= fff
                before_start_indexes = before_start_indexes[:, :factor, :]
                start_indexes = start_indexes[:, :factor, :]
                end_indexes = end_indexes[:, :factor, :]
                break

            before_start_indexes,start_indexes, end_indexes= fill_index(before_start_indexes,start_indexes, end_indexes,end_with_depot,select_index_next,length_sub,fff=fff)

            if  (start_indexes[:, :, 1] < 0).any():
                factor = fff
                before_start_indexes = before_start_indexes[:, :factor, :]
                start_indexes = start_indexes[:, :factor, :]
                end_indexes = end_indexes[:, :factor, :]
                break

        before_start_indexes[:,:,1] +=1
        start_indexes[:,:,1] +=1
        end_indexes[:,:,1] +=1

        before_start_index = start_indexes.clone()
        before_start_index = before_start_index[:,factor-1,:]
        before_start_index_ = select_before_start_depot_node(start_from_depot, before_start_index, start_index_1,temp6,index1)

        before_end_index_ = start_from_depot[before_start_index_]
        before_end_index_[:,:,1] = before_end_index_[:,:,1]
        before_start_indexes[:,[factor-1],:]= before_end_index_

        before_start_indexes = torch.flip(before_start_indexes,dims=[1])
        start_indexes = torch.flip(start_indexes, dims=[1])
        end_indexes = torch.flip(end_indexes, dims=[1])

        def cal_before_demands_sum(double_solution,problems,before_start_indexes_,start_indexes_,aug_batch_size,problems_size):
            before_start_indexes = before_start_indexes_.clone().reshape(aug_batch_size,-1)
            start_indexes = start_indexes_.clone().reshape(aug_batch_size,-1)

            index1 = torch.arange(problems_size)[None,:].repeat(aug_batch_size,1)
            x2 =  index1 >=before_start_indexes[:,1][:,None]
            x3 =  index1 <start_indexes[:,1][:, None]
            x4 = x2 * x3
            double_solution_demand = problems[:, :, 2][
                torch.arange(aug_batch_size)[:, None].repeat(1, problems_size), double_solution[:,:, 0]]
            double_solution_demand =double_solution_demand* x4
            before_demands_sum = torch.sum(double_solution_demand,dim=1)
            return before_demands_sum

        double_solution = torch.repeat_interleave(solution,repeats=factor,dim=0)
        problems = torch.repeat_interleave(problems,repeats=factor,dim=0)
        aug_batch_size = batch_size*factor

        before_demand = cal_before_demands_sum(double_solution,problems,before_start_indexes,start_indexes,aug_batch_size,problems_size)

        self.satisfy_demand = before_demand
        problems[:, :, 3] = problems[:, :, 3] - self.satisfy_demand[:, None]


        index = torch.arange(problems_size)[None, :].repeat(aug_batch_size, 1)
        first_node_index = start_indexes[:,:,1].reshape(aug_batch_size,1)
        end_node_index = end_indexes[:,:,1].reshape(aug_batch_size,1)
        index2 = index >= first_node_index
        index3 = index < end_node_index
        index4 = index2.int() * index3.int()

        new_sulution= double_solution[index4.gt(0.5)].reshape(aug_batch_size,length_sub,2)
        origin_sub_solution = new_sulution.clone()

        sub_solution_node = new_sulution[:, :, 0]



        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
        new_sulution[:, :, 0] = new_sulution_rank + 1

        index_2, _ =new_sulution_ascending.repeat(1,4).type(torch.long).sort(dim=-1, descending=False)

        index_1 = torch.arange(aug_batch_size, dtype=torch.long)[:, None].expand(aug_batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(aug_batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(aug_batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)

        return new_data, new_sulution,first_node_index,end_node_index,length_of_subpath,\
            double_solution,origin_sub_solution,index4,factor

    def decide_whether_to_repair_solution_V2(self,
                                          after_repair_sub_solution, before_reward, after_reward,
                                             double_solution,origin_sub_solution,index4,origin_batch_size,factor):

        aug_batch_size = double_solution.shape[0]

        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)
        index = torch.arange(jjj.shape[0])[:, None].repeat(1, jjj.shape[1])
        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]

        after_repair_sub_solution[:, :, 0] = kkk_2

        if_repair = before_reward > after_reward

        index4[if_repair] = index4[if_repair] * 2
        index5 = index4.reshape(origin_batch_size,factor,-1)
        index6 = torch.sum(index5,dim=1)

        index7 = torch.arange(start=0,end=aug_batch_size,step=factor)

        double_solution = double_solution[index7]

        double_solution[index6.gt(1.5).unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair.long().gt(0.5)].ravel()
        after_repair_complete_solution = double_solution


        return after_repair_complete_solution

    def destroy_solution_PRC(self, problem, complete_solution,length_sub,first_index_sub):


        self.problems, self.solution, first_node_index,end_node_index,length_of_subpath,\
            double_solution,origin_sub_solution,index4,factor = \
            self.sampling_subpaths_repair_V2(
            problem, complete_solution,length_sub,first_index_sub,mode=self.env_params['mode'])


        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution,
                                                              test_in_vrplib=self.env_params['test_in_vrplib'],
                                                              need_optimal=False)

        return partial_solution_length,first_node_index,end_node_index,length_of_subpath,\
            double_solution,origin_sub_solution,index4,factor

    def Rearrange_solution_clockwise(self, problem, solution):

        problem_size = solution.shape[1]
        coor = problem[:, :, [0, 1]].clone()
        order_node = solution[:, :, 0]
        order_flag = solution[:, :, 1]

        batch_size = solution.shape[0]

        visit_depot_num = torch.sum(order_flag, dim=1)

        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((order_flag, torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        start_from_depot2 = order_flag.nonzero()

        start_from_depot3 = order_flag.roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = order_node.repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)
        subtour_lengths = (sub_tours_padding > 1).int().sum(1)

        repeated_coor = torch.repeat_interleave(coor, repeats=visit_depot_num, dim=0)
        depot_coor = repeated_coor[:, [0], :].clone()
        repeated_coor[:, 0, :] = 0

        subtours_coor = repeated_coor.gather(dim=1, index=sub_tours_padding[:, :, None].repeat(1, 1, 2))
        subtours_coor = torch.cat((subtours_coor, depot_coor), dim=1)
        subtours_coor_sum = torch.sum(subtours_coor, dim=1)
        subtours_centroid = subtours_coor_sum / (subtour_lengths + 1)[:, None]
        subtours_centroid_total_num = subtours_centroid.shape[0]

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()
        temp_index = np.dot(visit_depot_num_numpy, temp_tri)

        temp_index_1 = torch.from_numpy(temp_index).long().cuda()

        temp_index_2 = visit_depot_num + temp_index_1

        x1 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) >= temp_index_1[:, None]
        x2 = torch.arange(subtours_centroid_total_num)[None, :].repeat(batch_size, 1) < temp_index_2[:, None]
        x3_ = (x1 * x2).int()
        x3 = x3_[:,:,None].repeat(1,1,2)

        subtours_centroid_repeat = subtours_centroid[None,:,:].repeat(batch_size,1,1)

        subtours_centroid_sperate = subtours_centroid_repeat * x3

        index2 = temp_index_1.clone().unsqueeze(1).unsqueeze(2).repeat(1,1,2)

        based_centroids = subtours_centroid_sperate.gather(dim=1,index=index2)

        single_depot_coor = coor[:, [0], :]

        repeated_depot_coor = coor[:, [0], :].repeat(1,all_subtour_num,1)

        all_centroid_depot_vectors = subtours_centroid_sperate - repeated_depot_coor

        based_centroid_depot_vectors = based_centroids - single_depot_coor

        repeated_based_centroid_depot_vectors = based_centroid_depot_vectors.repeat(1,all_subtour_num,1)

        x1_times_x2 = (repeated_based_centroid_depot_vectors * all_centroid_depot_vectors).sum(2)

        x1_module_length = torch.sqrt((repeated_based_centroid_depot_vectors**2).sum(2))
        x2_module_length = torch.sqrt((all_centroid_depot_vectors**2).sum(2))

        cos_value = x1_times_x2 / (x1_module_length*x2_module_length)
        cos_value[cos_value.ge(1)] =  1 - 1e-5
        cos_value[cos_value.le(-1)] = -1 + 1e-5
        cross_value = np.cross(repeated_based_centroid_depot_vectors.cpu().numpy(), all_centroid_depot_vectors.cpu().numpy())

        cross_value = torch.tensor(cross_value)
        negtivate_sign_2 = torch.ones(size=(cross_value.shape[0],cross_value.shape[1]))
        negtivate_sign_2[cross_value.lt(0)] = -1

        theta_value = torch.arccos(cos_value)
        theta_value = torch.where(torch.isnan(theta_value), torch.full_like(theta_value, 2 * 3.1415926), theta_value)
        theta_value = negtivate_sign_2*theta_value

        theta_value[theta_value.lt(0)] +=2 * 3.1415926

        theta_value[x3_.le(0)] = 6*3.1415926
        theta_value_sort_value, theta_value_sort_index = torch.sort(theta_value,dim=1)

        repeated_sub_tours_padding = sub_tours_padding.unsqueeze(0).repeat(batch_size,1,1)

        gather_theta_value_sort_index = theta_value_sort_index.unsqueeze(2).repeat(1,1,max_subtour_length)

        resort_repeated_sub_tours_padding = repeated_sub_tours_padding.gather(dim=1,index=gather_theta_value_sort_index)

        x4 = torch.arange(all_subtour_num)[None,:].repeat(batch_size,1)

        x5 = (x4 < visit_depot_num[:,None]).int()
        x6 = x5.unsqueeze(2).repeat(1,1,max_subtour_length)

        resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding*x6

        resort_repeated_sub_tours_padding = resort_repeated_sub_tours_padding.reshape(batch_size,-1)


        resort_sub_tours = resort_repeated_sub_tours_padding[resort_repeated_sub_tours_padding.gt(0)].reshape(batch_size,-1)

        repeated_sub_tours_length = sub_tours_length[sub_tours_length.gt(0)].unsqueeze(0).repeat(batch_size,1)

        resort_repeated_sub_tours_length = repeated_sub_tours_length.gather(dim=1,index= theta_value_sort_index)
        resort_repeated_sub_tours_length = resort_repeated_sub_tours_length*x5
        max_subtour_number = visit_depot_num.max()

        resort_repeated_sub_tours_length = resort_repeated_sub_tours_length[:,:max_subtour_number]

        temp_tri = np.triu(np.ones((batch_size,max_subtour_number.item(), max_subtour_number.item())), k=1)
        resort_repeated_sub_tours_length_numpy = resort_repeated_sub_tours_length.clone().cpu().numpy()
        temp_index = np.dot(resort_repeated_sub_tours_length_numpy, temp_tri)
        temp_index_1 = torch.from_numpy(temp_index).long().cuda()
        index1 = torch.arange(batch_size)
        temp_index_1 = temp_index_1[index1,index1]
        temp_index_1[temp_index_1.ge(problem_size)]=0


        flag = torch.zeros(size=(batch_size,problem_size),dtype=torch.int)
        index1 = torch.arange(batch_size)[:,None].repeat(1,max_subtour_number)


        flag[index1,temp_index_1]=1

        solution_ = torch.cat((resort_sub_tours.unsqueeze(2),flag.unsqueeze(2)),dim=2)
        return solution_


    def Rearrange_solution_caller(self, problem, solution):

        solution_tmp = solution.clone()
        problem_tmp = problem.clone()
        times = 2  # if OOM appears due to this method, change the value of 'times' to the batch size.
        batch_size = solution.shape[0]
        for i in range(times):
            begin_ = int(batch_size / times) * i
            end_ = int(batch_size / times) * (i + 1)
            solution_tmp[begin_:end_] = self.Rearrange_solution_clockwise(
                problem_tmp[begin_:end_], solution_tmp[begin_:end_])

        return solution_tmp


    def random_insert(self,origin_problem):
        from utils.insertion import cvrp_random_insertion

        print('random insertion begin!')
        initial_solution = []
        for kk in range(origin_problem.shape[0]):
            pos = origin_problem[kk, 1:, :2].clone().cpu().numpy()
            depotpos = origin_problem[kk, 0, :2].clone().cpu().numpy()
            demands = origin_problem[kk, 1:, 2].clone().cpu().numpy()
            capacity = origin_problem[kk, 0, 3].clone().cpu().numpy()
            capacity = int(capacity)

            route = cvrp_random_insertion(pos, depotpos, demands, capacity)
            solution = []
            for i in range(len(route)):
                sub_tour = (route[i] + 1).tolist()
                solution += [0]
                solution += sub_tour
                solution += [0]
            solution = torch.tensor(solution).reshape(1, -1)
            solution = self.tran_to_node_flag(solution)
            if initial_solution == []:
                initial_solution = solution
            else:
                initial_solution = torch.cat((initial_solution, solution), dim=0)
        return initial_solution

    def shuffle_data(self):

        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_capacity = self.raw_data_capacity[index]
        self.raw_data_demand = self.raw_data_demand[index]
        self.raw_data_cost = self.raw_data_cost[index]
        self.raw_data_node_flag = self.raw_data_node_flag[index]


    def load_raw_data(self,episode,problem_sizes=[100], begin_index=0,first_time_repair =False, repair = False,device=None):


        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])

            return tow_col_node_flag

        self.datas = {}
        if first_time_repair:
            print('First time repair')
            for i in range(len(episode)):
                instance ={}
                problem_size = problem_sizes[i]
                episode_ = episode[i]
                [instance['coor'],instance['demand'],instance['capacity']] = self.generate_nazari_vrp_data(episode_, problem_size)
                self.datas[str(problem_size) + '_' + str(episode_)] = instance

            self.make_dir(self.env_params['data_path_pt'][0])
            torch.save(self.datas, self.env_params['data_path_pt'][0]+self.env_params['data_path_pt'][1])

        else:
            if repair:
                print('Repair')
                if device is None:
                    self.datas = torch.load(self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][1])
                else:
                    self.datas = torch.load(self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][1],
                                            map_location=device)
            else:
                if self.test_in_vrplib:
                    print('Load vrplib test data')
                    self.cvrp_node_coords, self.cvrp_demands, self.cvrp_capacitys, \
                    self.vrplib_cost, self.vrplib_name = self.make_vrplib_data(self.vrplib_path, episode)


                else:

                    print('Load test data')

                    self.raw_data_nodes = []
                    self.raw_data_capacity = []
                    self.raw_data_demand = []
                    self.raw_data_cost = []
                    self.raw_data_node_flag = []
                    for line in tqdm(open(self.data_path, "r").readlines()[0 + begin_index:episode + begin_index],
                                     ascii=True):
                        line = line.split(",")

                        depot_index = int(line.index('depot'))
                        customer_index = int(line.index('customer'))
                        capacity_index = int(line.index('capacity'))
                        demand_index = int(line.index('demand'))
                        cost_index = int(line.index('cost'))
                        node_flag_index = int(line.index('node_flag'))

                        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                        customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                                    range(customer_index + 1, capacity_index, 2)]

                        loc = depot + customer

                        capacity = int(float(line[capacity_index + 1]))
                        # demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                        if int(line[demand_index + 1]) == 0:
                            demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                        else:
                            demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

                        cost = float(line[cost_index + 1])
                        node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

                        node_flag = tow_col_nodeflag(node_flag)

                        self.raw_data_nodes.append(loc)
                        self.raw_data_capacity.append(capacity)
                        self.raw_data_demand.append(demand)
                        self.raw_data_cost.append(cost)
                        self.raw_data_node_flag.append(node_flag)

                    self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
                    # shape (B,V+1,2)  customer num + depot
                    self.raw_data_capacity = torch.tensor(self.raw_data_capacity, requires_grad=False)
                    # shape (B )
                    self.raw_data_demand = torch.tensor(self.raw_data_demand, requires_grad=False)
                    # shape (B,V+1) customer num + depot
                    self.raw_data_cost = torch.tensor(self.raw_data_cost, requires_grad=False)
                    # shape (B )
                    self.raw_data_node_flag = torch.tensor(self.raw_data_node_flag, requires_grad=False)
                    # shape (B,V,2)

                print(f'load raw dataset done!', )

    def make_dataset(self, filename, episode, batch_size, num_samples):
        nodes_coords = []
        tour = []

        print('\nLoading from {}...'.format(filename))
        print(filename)

        for line in tqdm(open(filename, "r").readlines()[episode:episode + batch_size], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes_coords.append(
                [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
            )

            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            tour.append(tour_nodes)

        nodes_coords = torch.tensor(nodes_coords)
        tour = torch.tensor(tour)
        return nodes_coords, tour

    def make_vrplib_data(self, filename,episode):

        node_coords = []
        demands = []
        capacitys = []
        costs = []
        names = []



        for line in tqdm(open(filename, "r").readlines()[0:episode], ascii=True):
            line = line.split(", ")

            name_index = int(line.index('[\'name\''))
            depot_index = int(line.index('\'depot\''))
            customer_index = int(line.index('\'customer\''))
            capacity_index = int(line.index('\'capacity\''))
            demand_index = int(line.index('\'demand\''))
            cost_index = int(line.index('\'cost\''))

            depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
            customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                        range(customer_index + 1, demand_index, 2)]

            loc = depot + customer

            capacity = int(float(line[capacity_index + 1]))
            # demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
            demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]

            cost = float(line[cost_index + 1])

            node_coords.append(loc)
            demands.append(demand)
            capacitys.append(capacity)
            costs.append(cost)
            names.append(line[name_index+1][1:-1])

        node_coords = np.array(node_coords,dtype=object)
        demands = np.array(demands,dtype=object)
        capacitys = np.array(capacitys)
        costs = np.array(costs)
        names = np.array(names)

        return node_coords, demands, capacitys, costs, names



    def reset(self, mode, sample_size = 1):
        self.selected_count = 0


        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_teacher_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)

        self.step_state = Step_State(problems=self.problems)
        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student,selected_flag_teacher,selected_flag_student):


        self.selected_count += 1


        gather_index = selected[:, None, None].expand((len(selected), 1, 4)) # shape [B,1,4]

        is_depot = selected_flag_teacher==1
        self.problems[is_depot, :, 3] =  self.raw_data_capacity.ravel()[0].item()

        self.current_node_temp = self.problems.gather(index=gather_index, dim=1).squeeze(1)
        demands = self.current_node_temp[:,2]
        smaller_ = self.problems[:, 0, 3] < demands

        selected_flag_teacher[smaller_] = 1
        self.problems[smaller_, :, 3] =  self.raw_data_capacity.ravel()[0].item()

        self.problems[:,:,3] =  self.problems[:,:,3]- demands[:,None]

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)

        self.selected_teacher_flag = torch.cat((self.selected_teacher_flag, selected_flag_teacher[:, None]), dim=1)

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)

        self.selected_student_flag = torch.cat((self.selected_student_flag, selected_flag_student[:, None]), dim=1)

        done = (self.selected_count == self.problems.shape[1]-1)
        if done:
            reward, reward_student = self._get_travel_distance()  # note the minus sign!
        else:
            reward, reward_student = None, None

        return self.step_state, reward, reward_student, done

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    def drawPic_VRP(self, coor_, order_node_,order_flag_,name='xx', optimal_tour_=None):
        # coor: shape (V,2)
        # order_node_: shape (V)
        # order_flag_: shape (V)


        coor = coor_.clone().cpu().numpy()
        order_node =  order_node_.clone().cpu().numpy()
        order_flag = order_flag_.clone().cpu().numpy()

        tour = []
        for i in range(len(order_node)):
            if order_flag[i]==1:
                tour.append(0)
                tour.append(order_node[i])
            if order_flag[i]==0:
                tour.append(order_node[i])

        if optimal_tour_ is not None:
            optimal_tour = optimal_tour_.clone().cpu().numpy()
        arr_max = np.max(coor)
        arr_min = np.min(coor)
        arr = (coor - arr_min) / (arr_max - arr_min)

        fig, ax = plt.subplots(figsize=(20, 20))

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.axis('off')
        plt.scatter(arr[0, 0], arr[0, 1], color='red', linewidth=15,marker='v')

        col_counter = order_flag.sum()
        colors = plt.cm.turbo(np.linspace(0, 1, col_counter)) # turbo
        np.random.seed(123)
        np.random.shuffle(colors)


        count = -1
        for i in range(len(tour) - 1):
            if tour[i]==0:
                count+=1
            tour = np.array(tour, dtype=int)

            start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
            end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
            plt.plot(start, end, color=colors[count], linewidth=3)  # ,linestyle ="dashed"

            plt.scatter(arr[tour[i], 0], arr[tour[i], 1], color='gray', linewidth=2)
            plt.scatter(arr[tour[i+1], 0], arr[tour[i+1], 1], color='gray', linewidth=2)

            if optimal_tour_ is not None:
                tour_optimal = np.array(optimal_tour, dtype=int)

                start_optimal = [arr[tour_optimal[i], 0], arr[tour_optimal[i + 1], 0]]
                end_optimal = [arr[tour_optimal[i], 1], arr[tour_optimal[i + 1], 1]]
                plt.plot(start_optimal, end_optimal, color='green', linewidth=1)



        b = os.path.abspath(".")
        path = b+'/figure'
        self.make_dir(path)
        plt.savefig(path+f'/{name}.pdf',bbox_inches='tight', pad_inches=0)

    def cal_length(self, problems, order_node, order_flag):

        order_node_ = order_node.clone()

        order_flag_ = order_flag.clone()

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        problem_size = problems.shape[1] - 1

        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = ((order_loc - flag_loc) ** 2)

        order_flag_[:,0]=0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = ((roll_loc - flag_loc) ** 2)

        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length

    def _get_travel_distance(self):

        if self.decode_method == 'greedy' :

            if self.test_in_vrplib:
                travel_distances = self.vrplib_cost
                self.problems[:,:,:2] =  self.problems[:,:,:2] * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]

            else:
                problems = self.problems[:,:,[0,1]]
                order_node = self.solution[:,:,0]
                order_flag = self.solution[:,:,1]
                travel_distances = self.cal_length( problems, order_node, order_flag)

            problems = self.problems[:, :, [0, 1]]
            order_node = self.selected_student_list.clone()
            order_flag = self.selected_student_flag.clone()

            travel_distances_student = self.cal_length(problems, order_node, order_flag)

            return travel_distances, travel_distances_student



    def _get_travel_distance_2(self, problems_, solution_,test_in_vrplib = False,need_optimal =False ):

        if test_in_vrplib:
            if need_optimal:
                return self.vrplib_cost, self.vrplib_name
            else:
                problems = problems_[:, :, [0, 1]].clone() * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]
                order_node = solution_[:, :, 0].clone()
                order_flag = solution_[:, :, 1].clone()
                travel_distances = self.cal_length(problems, order_node, order_flag)
        else:
            problems = problems_[:, :, [0, 1]].clone()
            order_node = solution_[:, :, 0].clone()
            order_flag = solution_[:, :, 1].clone()
            travel_distances = self.cal_length(problems, order_node, order_flag)

        return travel_distances

    def destroy_solution(self, problem, complete_solution,sublength, first_node_index):


        self.problems, self.solution, first_node_index,length_of_subpath,double_solution = self.sampling_subpaths_repair(
            problem, complete_solution, mode=self.env_params['mode'],length_fix=sublength)


        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution,
                                                              test_in_vrplib=self.env_params['test_in_vrplib'],
                                                              need_optimal=False)

        return partial_solution_length,first_node_index,length_of_subpath,double_solution


    def sampling_subpaths_repair(self, problems, solution, length_fix=None, mode='test', repair=True):

        problems_size = problems.shape[1] - 1

        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        length_of_subpath = torch.randint(low=4, high=problems_size, size=[1])[0]  # in [4,N]
        if length_fix is not None:
            length_of_subpath=length_fix

        start_from_depot = solution[:, :, 1].nonzero()

        end_with_depot = start_from_depot
        end_with_depot[:, 1] = end_with_depot[:, 1] - 1
        end_with_depot[end_with_depot.le(-0.5)] = solution.shape[1] - 1

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        p = torch.rand(len(visit_depot_num))
        select_end_with_depot_node_index = p * visit_depot_num
        select_end_with_depot_node_index = torch.floor(select_end_with_depot_node_index).long()

        temp_tri = np.triu(np.ones((len(visit_depot_num), len(visit_depot_num))), k=1)
        visit_depot_num_numpy = visit_depot_num.clone().cpu().numpy()

        temp_index = np.dot(visit_depot_num_numpy, temp_tri)
        temp_index_torch = torch.from_numpy(temp_index).long().cuda()

        select_end_with_depot_node_index_ = select_end_with_depot_node_index + temp_index_torch

        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index_, 1]

        double_solution = torch.cat((solution, solution), dim=1)

        select_end_with_depot_node = select_end_with_depot_node + problems_size

        indexx = torch.arange(length_of_subpath).repeat(batch_size, 1)
        offset = select_end_with_depot_node - length_of_subpath + 1

        indexxxx = indexx + offset[:, None]


        sub_solu_index1 = torch.arange(batch_size)[:,None].repeat(1,2*length_of_subpath)
        sub_solu_index2 =indexxxx.repeat_interleave(2,dim=1)
        sub_solu_index3 = torch.arange(double_solution.shape[2])[None,:].repeat(batch_size,length_of_subpath)
        sub_solution = double_solution[sub_solu_index1,sub_solu_index2,sub_solu_index3].reshape(batch_size,length_of_subpath,2)

        offset_index = problems.shape[0]
        start_index = indexxxx[:, 0]


        x1 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) <= start_index[:offset_index][:, None]

        start_capacity = 0
        before_is_via_depot_all = double_solution[:offset_index, :, 1] * x1
        before_is_via_depot = before_is_via_depot_all.nonzero()

        visit_depot_num_2 = torch.sum(before_is_via_depot_all, dim=1)

        select_end_with_depot_node_index_2 = visit_depot_num_2 - 1

        temp_tri_2 = np.triu(np.ones((len(visit_depot_num_2), len(visit_depot_num_2))), k=1)
        visit_depot_num_numpy_2 = visit_depot_num_2.clone().cpu().numpy()

        temp_index_2 = np.dot(visit_depot_num_numpy_2, temp_tri_2)
        temp_index_torch_2 = torch.from_numpy(temp_index_2).long().cuda()

        select_end_with_depot_node_index_2 = select_end_with_depot_node_index_2 + temp_index_torch_2
        before_is_via_depot_index = before_is_via_depot[select_end_with_depot_node_index_2]

        before_start_index = before_is_via_depot_index[:, 1]
        x2 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) < start_index[:offset_index][:, None]
        x3 = torch.arange(double_solution[:offset_index, :, 1].shape[1]) >= before_start_index[:, None]
        x4 = x2 * x3
        double_solution_demand = problems[:offset_index, :, 2][
            torch.arange(offset_index)[:, None].repeat(1, double_solution.shape[1]), double_solution[:offset_index, :, 0]]

        before_demand = double_solution_demand * x4

        self.satisfy_demand = before_demand.sum(1)

        problems[:offset_index, :, 3] = problems[:offset_index, :, 3] - self.satisfy_demand[:, None]

        sub_solution_node = sub_solution[:, :, 0]


        new_sulution_ascending, rank = torch.sort(sub_solution_node, dim=-1, descending=False)  # 升序
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
        sub_solution[:, :, 0] = new_sulution_rank + 1

        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending, new_sulution_ascending, new_sulution_ascending), dim=1). \
            type(torch.long).sort(dim=-1, descending=False)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, embedding_size)
        new_data = torch.cat((problems[:, 0, :].unsqueeze(dim=1), new_data), dim=1)


        if repair == True:
            return new_data, sub_solution,start_index,length_of_subpath,double_solution
        else:
            return new_data, sub_solution


    def decide_whether_to_repair_solution(self, after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):


        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)

        temp = torch.arange(double_solution.shape[1])

        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)

        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)

        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]


        after_repair_sub_solution[:, :, 0] = kkk_2

        if_repair = before_reward > after_reward

        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        double_solution[if_repair] = need_to_repari_double_solution

        x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()

        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()

        x8 = x6 * x7

        after_repair_complete_solution = double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)

        return after_repair_complete_solution

    def tran_to_node_flag(self, node_list):
        '''
        :param node_list: [B, V+n]
        :return: [B, V, 2]
        '''

        batch_size = node_list.shape[0]

        index_smaller_0_shift = torch.roll(torch.le(node_list, 0), shifts=1, dims=1).long()
        index_smaller_0_shift[:,0]=0
        index_bigger_0 = torch.gt(node_list, 0).long()

        flag_index = index_smaller_0_shift * index_bigger_0

        save_index = torch.gt(node_list, 0.1)

        save_node = node_list[save_index].reshape(batch_size, -1)
        save_flag = flag_index[save_index].reshape(batch_size, -1)

        node_flag_1 = torch.cat((save_node.unsqueeze(2), save_flag.unsqueeze(2)), dim=2)

        return node_flag_1

    def valida_solution_legal(self, problem, solution,capacity_=50):

        capacity = capacity_

        demand = problem[:, :, 2]

        order_node = solution[:, :, 0].clone()

        uniques = torch.unique(order_node[0])
        if len(uniques) != problem.shape[1] - 1:
            assert False, 'wrong node list!'

        batch_size = solution.shape[0]

        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

        index = torch.arange(sub_tours_padding.shape[0])[:, None].repeat(1, sub_tours_padding.shape[1])
        sub_tours_demands = demands[index, sub_tours_padding].sum(dim=1)
        if_legal = (sub_tours_demands > capacity)

        if if_legal.any():
            assert False, 'wrong capacity!'

        return

    def generate_nazari_vrp_data(self,dataset_size, vrp_size):
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.,
            200: 80.,
            500: 100.,
            1000: 250.,
            5000: 500.,
            10000: 1000.,
            50000: 2000.,
            100000: 2000.,
        }
        return torch.tensor(np.random.uniform(size=(dataset_size, vrp_size + 1, 2)),dtype=torch.float32), \
               torch.tensor(np.concatenate(
                   (np.zeros((dataset_size, 1), dtype=int), np.random.randint(1, 10, size=(dataset_size, vrp_size))),
                   axis=1)), \
               torch.tensor(np.full(dataset_size, CAPACITIES[vrp_size]),dtype=torch.float32)

