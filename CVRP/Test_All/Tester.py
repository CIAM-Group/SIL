import torch

import os
from logging import getLogger

from CVRP.Test_All.VRPEnv import VRPEnv as Env
from CVRP.Test_All.VRPModel import VRPModel as Model

from utils.utils import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
class VRPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        seed = 123
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuda
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

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname,map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        if self.env_params['load_way'] == 'allin':
            self.env.load_raw_data(self.tester_params['test_episodes'])

        k_nearest = self.env_params['k_nearest']
        beam_width = self.env_params['beam_width']
        decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        problems_le_7000 = []
        problems_gt_7000 = []
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score, problems_size = self._test_one_batch(
                episode, batch_size, k_nearest, decode_method, clock=self.time_estimator_2,logger = self.logger)
            if self.env_params['vrplib_path']:
                if problems_size <= 7000:
                    problems_le_7000.append((score_student_mean - score) / score)
                elif 7000 < problems_size:
                    problems_gt_7000.append((score_student_mean - score) / score)

                print('problems_le_7000 mean gap:', np.mean(problems_le_7000), len(problems_le_7000))
                print('problems_gt_7000 mean gap:', np.mean(problems_gt_7000), len(problems_gt_7000))

            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, score_student_mean, aug_score))

            all_done = (episode == test_num_episode)

            gap_ = 1
            if all_done and not self.env_params['vrplib_path']:
                self.logger.info(" *** Test_All Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
                # self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg - score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                          before_complete_solution, before_repair_sub_solution,
                                          after_repair_sub_solution, before_reward, after_reward,
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

        kkk_1 = jjj[index, before_repair_sub_solution[:, :, 0] - 1]

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

    def _test_one_batch(self, episode, batch_size, k_nearest, decode_method, clock=None,logger = None):

        random_seed = 123
        torch.manual_seed(random_seed)


        self.model.eval()

        max_memory_allocated_before = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024
        print('max_memory_allocated before',max_memory_allocated_before,'MB')
        torch.cuda.reset_peak_memory_stats(device=self.device)


        with torch.no_grad():

            self.env.load_problems(episode, batch_size, only_test=True)

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            self.origin_problem = self.env.problems.clone().detach()


            if self.env.test_in_vrplib:
                optimal_length_and_names = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                           test_in_vrplib=self.env.test_in_vrplib,
                                                                           need_optimal=self.env.test_in_vrplib)
                self.optimal_length = optimal_length_and_names[0][episode]
                self.vrp_names = optimal_length_and_names[1][episode]

            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            if self.env_params['random_insertion']:
                from utils.insertion import cvrp_random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()

                print('random insertion begin!')
                initial_solution = []
                for kk in range(self.origin_problem.shape[0]):
                    pos = self.origin_problem[kk,1:,:2].clone().cpu().numpy()
                    depotpos = self.origin_problem[kk,0,:2].clone().cpu().numpy()
                    demands = self.origin_problem[kk,1:,2].clone().cpu().numpy()
                    capacity = self.origin_problem[kk,0,3].clone().cpu().numpy()
                    capacity = int(capacity)

                    route = cvrp_random_insertion(pos, depotpos, demands, capacity)
                    solution = []
                    for i in range(len(route)):
                        sub_tour = (route[i] + 1).tolist()
                        solution += [0]
                        solution += sub_tour
                        solution += [0]

                    solution=torch.tensor(solution).reshape(1,-1)
                    solution = self.env.tran_to_node_flag(solution)
                    if initial_solution==[]:
                        initial_solution = solution
                    else:
                        initial_solution = torch.cat((initial_solution,solution),dim=0)

                best_select_node_list = initial_solution
            else:

                B_V = batch_size * 1
                from tqdm import tqdm
                with tqdm(total=self.env.problem_size) as pbar:
                    while not done:
                        pbar.update(1)
                        loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity, decode_method=decode_method)
                        if current_step == 0:
                            selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                            selected_flag_student = selected_flag_teacher
                        current_step += 1

                        state, reward, reward_student, done = \
                            self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)
                        # print(current_step)


                print('Get first complete solution!')

                best_select_node_list = torch.cat((self.env.selected_student_list.reshape(batch_size, -1, 1),
                                                   self.env.selected_student_flag.reshape(batch_size, -1, 1)), dim=2)

            max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                  test_in_vrplib=self.env.test_in_vrplib)

            escape_time, _ = clock.get_est_string(1, 1)


            if self.env.test_in_vrplib:
                self.logger.info("curr00, {} gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(self.vrp_names,
                  (( current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                 escape_time, current_best_length.mean().item(),  self.optimal_length.mean().item()))
            else:
                self.logger.info("curr00,  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB,".format(
                    ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                current_best_length.mean().item(), self.optimal_length.mean().item(), max_memory_allocated_after))


            budget = self.env_params['budget']

            origin_problem_size = self.origin_problem.shape[1]
            origin_batch_size = batch_size
            length_all  = torch.randint(4,
                                        high=self.env_params['repair_max_sub_length']+1, size=[budget])  # in [4,N]
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])  # in [4,N]


            for bbbb in range(budget):
                torch.cuda.empty_cache()

                for i in range(batch_size):
                    best_select_node_list[ i: (i + 1)] = self.env.Rearrange_solution_clockwise(
                                                            self.origin_problem[ i:(i + 1)],
                                                            best_select_node_list[ i:(i + 1)])

                self.env.load_problems(episode, batch_size, only_test=True)

                best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)
                fix_length =torch.randint(low=4, high=self.env_params['repair_max_sub_length'], size=[1])[0]  # in [4,N]
                if self.env_params['PRC']:
                    partial_solution_length, first_node_index, end_node_index, length_of_subpath, \
                    double_solution, origin_sub_solution, index4, factor = \
                        self.env.destroy_solution_PRC(self.env.problems, best_select_node_list,
                                                  length_all[bbbb], first_index_all[bbbb])
                else:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                    self.env.destroy_solution(self.env.problems, best_select_node_list,length_all[bbbb], first_index_all[bbbb])

                before_repair_sub_solution = self.env.solution

                self.env.batch_size = before_repair_sub_solution.shape[0]

                before_reward = partial_solution_length

                current_step = 0

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node


                while not done:
                    if current_step == 0:

                        selected_teacher = self.env.solution[:, 0, 0]
                        selected_flag_teacher = self.env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher


                    else:
                        _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity, decode_method=decode_method)

                    current_step += 1

                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

                ahter_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                       self.env.selected_student_flag.unsqueeze(2)), dim=2)

                after_reward = reward_student

                if self.env_params['PRC']:
                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution_V2(
                        ahter_repair_sub_solution,before_reward, after_reward,double_solution, origin_sub_solution,
                        index4, origin_batch_size, factor)
                else:

                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution(ahter_repair_sub_solution,
                        before_reward, after_reward, first_node_index, length_of_subpath,double_solution)
                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                      test_in_vrplib=self.env.test_in_vrplib)

                escape_time, _ = clock.get_est_string(1, 1)

                max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

                if self.env.test_in_vrplib:
                    self.logger.info( " step{},{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                            bbbb, self.vrp_names, ((  current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                            escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
                else:
                    self.logger.info(
                        " step{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f},, Memory:{:4f}MB".format(
                            bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                            escape_time, current_best_length.mean().item(), self.optimal_length.mean().item(),
                            max_memory_allocated_after))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                  test_in_vrplib=self.env.test_in_vrplib)

            print(f'current_best_length', (current_best_length.mean() - self.optimal_length.mean())
                  / self.optimal_length.mean() * 100, '%', 'escape time:', escape_time,
                  f'optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}')

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.optimal_length.mean().item(), self.env.problem_size
