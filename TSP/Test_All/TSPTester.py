
import torch

import os
from logging import getLogger

from TSP.Test_All.TSPEnv import TSPEnv as Env
from TSP.Test_All.TSPModel import TSPModel as Model

from utils.utils import *
import random
class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,):

        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        seed = 123
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)

        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))


    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()


        self.env.load_raw_data(self.tester_params['test_episodes'] )

        k_nearest = self.env_params['k_nearest']

        decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_le_5000 = []
        problems_gt_5000 = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score,problems_size = self._test_one_batch(episode,batch_size,k_nearest,decode_method,clock=self.time_estimator_2)

            print('max_memory_allocated',torch.cuda.max_memory_allocated(device=self.device ) / 1024 / 1024,'MB')

            if self.env.test_in_tsplib:
                if problems_size <= 1000:
                    problems_le_5000.append((score_student_mean - score) / score)
                elif 5000 < problems_size:
                    problems_gt_5000.append((score_student_mean - score) / score)

                print('problems_le_5000 mean gap:', np.mean(problems_le_5000), len(problems_le_5000))
                print('problems_gt_5000 mean gap:', np.mean(problems_gt_5000), len(problems_gt_5000))

            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score,score_student_mean, aug_score))

            all_done = (episode == test_num_episode)

            if all_done and not self.env.test_in_tsplib:
                self.logger.info(" *** Test_All Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))

                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100



        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                            before_complete_solution, before_repair_sub_solution,
                                          after_repair_sub_solution,before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size  = int(double_solution.shape[1]/2)


        other_part_1 = double_solution[:,:first_node_index]
        other_part_2 = double_solution[:,first_node_index+length_of_subpath:]
        origin_sub_solution = double_solution[:, first_node_index : first_node_index+length_of_subpath]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)

        index = torch.arange(jjj.shape[0])[:,None].repeat(1,jjj.shape[1])

        kkk_2 = jjj[index,after_repair_sub_solution]

        if_repair = before_reward>after_reward

        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                        kkk_2[if_repair],
                                                        other_part_2[if_repair]),dim=1)
        after_repair_complete_solution = double_solution[:,first_node_index:first_node_index+the_whole_problem_size]



        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size,k_nearest,decode_method,clock=None):

        self.model.eval()


        max_memory_allocated_before = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

        print('max_memory_allocated before', max_memory_allocated_before, 'MB')

        torch.cuda.reset_peak_memory_stats(device=self.device)

        with torch.no_grad():

            self.env.load_problems(episode, batch_size,only_test=True)
            self.origin_problem = self.env.problems
            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            if self.env.test_in_tsplib:
                optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                       test_in_tsplib=self.env.test_in_tsplib,
                                                                       need_optimal=True)
                self.optimal_length = optimal_length[episode]
                self.name = name[episode]

            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            IF_random_insertion = self.env_params['random_insertion']

            if IF_random_insertion:
                from utils.insertion import random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()
                problem_size = dataset.shape[1]
                width = 1
                print('random insertion begin!')
                orders = [torch.randperm(problem_size) for i in range(width)]
                pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for
                          instance in
                          dataset]
                pi_all = np.array(pi_all, dtype=np.int64)
                best_select_node_list = torch.tensor(pi_all)

            else:

                B_V = batch_size * 1
                current_step = 0
                state, reward, reward_student, done = self.env.pre_step()
                from tqdm import tqdm
                with tqdm(total=self.env.problem_size) as pbar:
                    while not done:
                        pbar.update(1)

                        if current_step == 0:
                            selected_teacher= torch.zeros(B_V,dtype=torch.int64)
                            selected_student = selected_teacher

                        else:
                            selected_teacher, _,_,selected_student = self.model(
                                state,self.env.selected_node_list,self.env.solution,current_step,
                                decode_method=decode_method)

                        current_step += 1

                        state, reward,reward_student, done = self.env.step(selected_teacher, selected_student)


                best_select_node_list = self.env.selected_node_list


            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                  test_in_tsplib=self.env.test_in_tsplib)

            max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024


            escape_time, _ = clock.get_est_string(1, 1)

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100

            if self.env.test_in_tsplib:
                self.logger.info("greedy, name:{}, gap:{:6f} %,  Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                    self.name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
            else:
                self.logger.info("curr00,  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                    gap, escape_time,  current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after ))

            budget = self.env_params['budget']


            origin_problem_size = self.origin_problem.shape[1]

            torch.save(best_select_node_list, f'TSP{origin_problem_size}_step0.pt')

            origin_batch_size = batch_size

            repair_max_sub_length = self.env_params['repair_max_sub_length']
            repair_max_sub_length = min(origin_problem_size, repair_max_sub_length)

            length_all = torch.randint(low=4, high=repair_max_sub_length + 1, size=[budget])
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])


            for bbbb in range(budget):

                self.env.problems = self.origin_problem.clone().detach()

                best_select_node_list = self.env.random_inverse_solution(best_select_node_list)


                if_PRC = self.env_params['PRC']

                if if_PRC:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution, \
                    origin_sub_solution, index4, factor = \
                        self.env.destroy_solution_PRC(self.env.problems, best_select_node_list, length_all[bbbb],
                                                      first_index_all[bbbb], )
                else:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                        self.env.destroy_solution(self.env.problems, best_select_node_list)

                before_reward = partial_solution_length

                before_repair_sub_solution = self.env.solution

                self.env.batch_size = before_repair_sub_solution.shape[0]

                current_step = 0

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()

                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, -1]
                        selected_student = self.env.solution[:, -1]

                    elif current_step == 1:
                        selected_teacher = self.env.solution[:, 0]
                        selected_student = self.env.solution[:, 0]

                    else:
                        selected_teacher, _,_,selected_student = self.model(
                            state,self.env.selected_node_list,self.env.solution,current_step,
                            decode_method=decode_method,repair = True)

                    current_step += 1
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

                ahter_repair_sub_solution = torch.roll(self.env.selected_node_list,shifts=-1,dims=1)

                after_reward = reward_student

                if if_PRC:
                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution_PRC(
                        ahter_repair_sub_solution, before_reward, after_reward, double_solution,
                        origin_batch_size, origin_sub_solution, index4, factor)
                else:
                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution(
                        ahter_repair_sub_solution, before_reward, after_reward,
                        first_node_index, length_of_subpath, double_solution)

                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(self.origin_problem,
                                                                      best_select_node_list, test_in_tsplib=self.env.test_in_tsplib)
                escape_time,_ = clock.get_est_string(1, 1)

                max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

                gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100

                if self.env.test_in_tsplib:

                    self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                            bbbb, self.name, gap, escape_time, current_best_length.mean().item(),
                            self.optimal_length.mean().item()))
                else:
                    self.logger.info("step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB".format(
                       bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                    current_best_length.mean().item(), self.optimal_length.mean().item(),max_memory_allocated_after))



        return self.optimal_length.mean().item(),current_best_length.mean().item(), current_best_length.mean().item(),self.env.problem_size
