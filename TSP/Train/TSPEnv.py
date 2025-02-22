from dataclasses import dataclass
import torch


from tqdm import tqdm
import numpy as np

import os

@dataclass
class Reset_State:
    problems: torch.Tensor


@dataclass
class Step_State:
    data: torch.Tensor
    first_node: torch.Tensor
    current_node: torch.Tensor


class TSPEnv:
    def __init__(self, **env_params):


        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params['data_path']

        self.batch_size = None

        self.problems = None
        self.first_node = None

        self.raw_data_nodes = []
        self.raw_data_tours = []

        self.selected_count = None
        self.current_node = None

        self.selected_node_list = None
        self.selected_student_list = None

        self.test_in_tsplib = env_params['test_in_tsplib']
        self.tsplib_path = env_params['tsplib_path']
        self.tsplib_cost = None
        self.tsplib_name = None
        self.tsplib_problems = None
        self.problem_max_min = None
        self.episode = None

        self.datas = {}


    def load_problems(self, episode, batch_size,problem_size_type=100, dataset_size = None,
                      current_best_solution=None, only_test = False, fix_length = None, sub_path=False):
        self.episode = episode
        self.batch_size = batch_size

        if not only_test:
            if dataset_size is not None:

                self.problems = self.datas[str(problem_size_type)+'_'+str(dataset_size)][episode:episode + batch_size]
                self.solution = torch.arange(problem_size_type, dtype=torch.int64)[None, :].repeat(batch_size, 1)

            if current_best_solution is not None:
                self.solution = current_best_solution
        else:
            self.problems, self.solution = self.raw_data_nodes[episode:episode + batch_size], \
                                            self.raw_data_tours[ episode:episode + batch_size]
        if sub_path:
            self.problems, self.solution = self.sampling_subpaths(self.problems, self.solution,mode='train',
                                                                  fix_length = fix_length)

        self.solution = self.random_inverse_solution(self.solution)

        if self.env_params['mode'] == 'test' and self.test_in_tsplib:
            self.tsplib_problems, self.tsplib_cost, self.tsplib_name = self.make_tsplib_data(self.tsplib_path)

            self.problems = torch.from_numpy(self.tsplib_problems[episode].reshape(1,-1,2)).cuda().float()
            self.problem_max_min = [torch.max(self.problems),torch.min(self.problems)]
            self.problems = (self.problems - self.problem_max_min[1])/(self.problem_max_min[0]-self.problem_max_min[1])
            self.solution = None
        self.problem_size = self.problems.shape[1]




    def sampling_subpaths(self, problems, solution, mode='test', repair=False, fix_length=None):
        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        first_node_index = torch.randint(low=0, high=problems_size, size=[1])[0]


        length_of_subpath = torch.randint(low=4, high=problems_size + 1, size=[1])[0]


        if fix_length is not None:
            length_of_subpath = fix_length

        double_solution = torch.cat([solution, solution], dim=-1)
        new_sulution = double_solution[:, first_node_index: first_node_index + length_of_subpath]

        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)

        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending), dim=1).type(torch.long).sort(dim=-1,
                                                                                                              descending=False)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size, embedding_size)
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, 2)

        if repair == True:
            return new_data, new_sulution_rank, first_node_index, length_of_subpath, double_solution
        else:
            return new_data, new_sulution_rank


    def shuffle_data(self):
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_tours = self.raw_data_tours[index]
        return index

    def load_raw_data(self, episode,problem_sizes=[100], begin_index=0,first_time_repair =False, repair = False):

        print('load raw dataset begin!')
        self.datas = {}
        if first_time_repair:
            for i in range(len(episode)):
                problem_size = problem_sizes[i]
                episode_ = episode[i]
                self.datas[str(problem_size)+'_'+str(episode_)] = torch.rand(size=(episode_, problem_size, 2))

            self.make_dir(self.env_params['data_path_pt'][0])
            torch.save(self.datas, self.env_params['data_path_pt'][0]+self.env_params['data_path_pt'][1])

        else:
            if repair:
                print('-------------------------- load --------------------------------')
                self.datas = torch.load(self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][1],map_location='cuda')

            else:
                print('load raw dataset begin!')

                self.raw_data_nodes = []
                self.raw_data_tours = []
                print(self.data_path)
                for line in tqdm(open(self.data_path, "r").readlines()[0 + begin_index:episode + begin_index],
                                 ascii=True):
                    line = line.split(" ")
                    num_nodes = int(line.index('output') // 2)
                    nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

                    self.raw_data_nodes.append(nodes)
                    tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

                    self.raw_data_tours.append(tour_nodes)

                self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
                self.raw_data_tours = torch.tensor(self.raw_data_tours, requires_grad=False)


        print(f'load raw dataset done!', )


    def make_tsplib_data(self, filename):
        instance_data = []
        cost = []
        instance_name = []

        for line in open(filename, "r").readlines():
            line = line.rstrip("\n")
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('\'', '')
            line = line.split(sep=',')

            line_data = np.array(line[2:], dtype=float).reshape(-1, 2)
            instance_data.append(line_data)
            cost.append(np.array(line[1], dtype=float))
            instance_name.append(np.array(line[0], dtype=str))
        instance_data = np.array(instance_data,dtype=object)
        cost = np.array(cost)
        instance_name = np.array(instance_name)

        return instance_data, cost, instance_name


    def destroy_solution(self, problem, complete_solution, fix_length):


        self.problems, self.solution,first_node_index,length_of_subpath,double_solution = self.sampling_subpaths(
            problem, complete_solution, mode='test',repair=True,fix_length=fix_length)

        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution,test_in_tsplib=self.env_params['test_in_tsplib'],
                                                                      need_optimal=False)
        return partial_solution_length,first_node_index,length_of_subpath,double_solution


    def destroy_solution_PRC(self, problem, complete_solution, length_sub, first_index_sub):

        self.problems, self.solution, first_node_index, length_of_subpath, double_solution, origin_sub_solution, index4, factor = \
            self.sampling_subpaths_PRC(
                problem, complete_solution, length_sub, first_index_sub)

        partial_solution_length = self._get_travel_distance_2(self.problems, self.solution,
                                                              test_in_tsplib=self.env_params['test_in_tsplib'],
                                                              need_optimal=False)
        return partial_solution_length, first_node_index, length_of_subpath, double_solution, origin_sub_solution, index4, factor

    def sampling_subpaths_PRC(self, problems, solution, length_sub, first_index_sub):

        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]

        first_node_index = first_index_sub

        length_of_subpath = length_sub

        factor = int(problems_size / length_of_subpath)

        # to avoid out of memory
        if factor > 250:
            factor = 250

        interval = torch.arange(factor) * length_of_subpath.item()

        solution = torch.roll(solution, shifts=-first_node_index.item(), dims=1)

        if factor > 1:
            solution = torch.repeat_interleave(solution, repeats=factor,
                                               dim=0)
            problems = torch.repeat_interleave(problems, repeats=factor, dim=0)

        first_node_index = interval[:, None].repeat(batch_size, 1)

        batch_size = solution.shape[0]

        index = torch.arange(problems_size)[None, :].repeat(batch_size, 1)

        index2 = index >= first_node_index
        index3 = index < first_node_index + length_of_subpath
        index4 = index2.long() * index3.long()

        new_sulution = solution[index4.gt(0.5)].reshape(batch_size, -1)

        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)

        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending), dim=1).type(torch.long).sort(dim=-1,
                                                                                                              descending=False)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[
            1])
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size,
                                                                                embedding_size)
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, 2)

        return new_data, new_sulution_rank, first_node_index, length_of_subpath, solution, new_sulution, index4, factor

    def decide_whether_to_repair_solution_PRC(self, after_repair_sub_solution, before_reward, after_reward,
                                              double_solution, origin_batch_size, origin_sub_solution, index4, factor):

        the_whole_problem_size = double_solution.shape[1]
        batch_size = double_solution.shape[0]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)

        index = torch.arange(jjj.shape[0])[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution]

        if_repair = before_reward > after_reward

        index4[if_repair] = index4[if_repair] * 2
        index5 = index4.reshape(origin_batch_size, factor, -1)
        index6 = torch.sum(index5, dim=1)

        index7 = torch.arange(start=0, end=batch_size, step=factor)
        double_solution = double_solution[index7]
        double_solution[index6.gt(1.5)] = kkk_2[if_repair.long().gt(0.5)].ravel()

        after_repair_complete_solution = double_solution[:, :the_whole_problem_size]

        return after_repair_complete_solution

    def reset(self):
        self.selected_count = 0

        self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

        self.step_state = Step_State(data=self.problems, first_node=self.first_node,
                                     current_node=self.current_node)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student, use_bs=False, beam_select_list=None, beam_width=None):

        self.selected_count += 1


        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)

        done = (self.selected_count == self.problems.shape[1])
        if done:
            reward, reward_student = self._get_travel_distance()
        else:
            reward, reward_student = None, None

        return self.step_state, reward, reward_student, done

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    def random_inverse_solution(self,solution):
        if_inverse = True
        if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]
        if if_inverse_index < 50:
            if_inverse = False

        if if_inverse:
            solution = torch.flip(solution, dims=[1])
        return solution

    def _get_travel_distance(self):

        if self.test_in_tsplib:
            travel_distances = self.tsplib_cost
            self.problems =  self.problems * (self.problem_max_min[0] - self.problem_max_min[1]) + self.problem_max_min[1]

        else:

            gathering_index = self.solution.unsqueeze(2).expand(self.batch_size, self.problems.shape[1], 2)

            seq_expanded = self.problems

            ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2)

            segment_lengths = segment_lengths.sum(2).sqrt()

            travel_distances = segment_lengths.sum(1)

        gathering_index_student = self.selected_student_list.unsqueeze(2).expand(-1, self.problems.shape[1], 2)
        ordered_seq_student = self.problems.gather(dim=1, index=gathering_index_student)
        rolled_seq_student = ordered_seq_student.roll(dims=1, shifts=-1)
        segment_lengths_student = ((ordered_seq_student - rolled_seq_student) ** 2)
        segment_lengths_student = segment_lengths_student.sum(2).sqrt()

        travel_distances_student = segment_lengths_student.sum(1)

        return travel_distances, travel_distances_student

    def _get_travel_distance_2(self, problems, solution,test_in_tsplib =False, need_optimal = False):

        if test_in_tsplib:
            if need_optimal:
                return self.tsplib_cost,self.tsplib_name
            else:
                problems_copy = problems.clone().detach() * (self.problem_max_min[0] - self.problem_max_min[1]) + \
                                self.problem_max_min[1]

                gathering_index = solution.unsqueeze(2).expand(problems_copy.shape[0], problems_copy.shape[1], 2)

                seq_expanded = problems_copy

                ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

                rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

                segment_lengths = ((ordered_seq - rolled_seq) ** 2)

                segment_lengths = segment_lengths.sum(2).sqrt()

                travel_distances = segment_lengths.sum(1)

                return travel_distances
        else:

            gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)

            seq_expanded = problems

            ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

            segment_lengths = ((ordered_seq - rolled_seq) ** 2)

            segment_lengths = segment_lengths.sum(2).sqrt()

            travel_distances = segment_lengths.sum(1)

        return travel_distances

    def decide_whether_to_repair_solution(self,
                                          after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size = int(double_solution.shape[1] / 2)

        other_part_1 = double_solution[:, :first_node_index]
        other_part_2 = double_solution[:, first_node_index + length_of_subpath:]
        origin_sub_solution = double_solution[:, first_node_index: first_node_index + length_of_subpath]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)

        index = torch.arange(jjj.shape[0])[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution]

        if_repair = before_reward > after_reward

        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                kkk_2[if_repair],
                                                other_part_2[if_repair]), dim=1)
        after_repair_complete_solution = double_solution[:, first_node_index:first_node_index + the_whole_problem_size]

        return after_repair_complete_solution