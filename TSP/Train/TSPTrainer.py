import numpy as np
import torch

import os
from logging import getLogger

from TSP.Train.TSPEnv import TSPEnv as Env
from TSP.Train.TSPModel import TSPModel as Model
from utils.utils import *

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch.nn as nn
import random

class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 trainer_params,
                 optimizer_params):

        self.env_params = env_params
        self.greedy_test_env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

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

        self.greedy_test_env_params['data_path'] = self.tester_params['data_path']
        self.greedy_test_env = Env(**self.greedy_test_env_params)

        self.model = Model(**self.model_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1

        self.RI_initial = self.env_params['RI_initial']

        if self.RI_initial:
            self.save_best_model(1)
        else:
            # Restore
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded !!')

        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        self.first_time_repair = self.trainer_params['first_time_repair']
        self.repair_before_train = self.trainer_params['repair_before_train']
        greedy_test_episodes = self.tester_params['episodes']
        greedy_test_batch = self.tester_params['batch_size']

        improve_iterations = self.trainer_params['improve_iterations']
        epoch_itervel = self.trainer_params['epoch_itervel']

        self.episodes = np.array(self.trainer_params['episodes'])
        self.problem_sizes = np.array(self.trainer_params['problem_sizes'])
        self.type_number = len(self.episodes)


        self.greedy_test_env.load_raw_data(greedy_test_episodes)

        save_gap = []


        score_optimal, score_student, gap = self.greedy_test_one_batch(greedy_test_episodes, greedy_test_batch)
        print('test gap:', gap)
        save_gap.append([self.start_epoch - 1, score_optimal, score_student, gap])

        self.env.load_raw_data(self.episodes, problem_sizes=self.problem_sizes,
                               first_time_repair=self.first_time_repair, repair=True)

        self.generate_best_student_list()


        if self.RI_initial:

            best_epoch_result_folder = [1,self.result_folder]
            current_epoch_result_folder = [1,self.result_folder]

        else:

            best_epoch_result_folder = [self.tester_params['model_load']['epoch'],
                                        self.tester_params['model_load']['path']]

            current_epoch_result_folder = [self.tester_params['model_load']['epoch'],
                                        self.tester_params['model_load']['path']]

        # loop1：
        for II in range(improve_iterations):


            if (not self.repair_before_train) and (not self.first_time_repair):
                if II ==0:
                    pass
                else:

                    self.repair_all_data(best_epoch_result_folder)
            else:
                self.repair_all_data(best_epoch_result_folder)


            end_epoch = self.trainer_params['epochs']

            self.start_epoch = best_epoch_result_folder[0] + 1

            # loop3：

            for epoch in range(self.start_epoch, self.start_epoch + epoch_itervel):

                self.suffle_data()

                train_score, train_student_score, train_loss = self.train_one_epoch(epoch)

                self.result_log.append('train_score', epoch, train_score)
                self.result_log.append('train_student_score', epoch, train_student_score)
                self.result_log.append('train_loss', epoch, train_loss)
                self.scheduler.step()


                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, end_epoch)
                self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch, end_epoch, elapsed_time_str, remain_time_str))

                all_done = (epoch == end_epoch)
                model_save_interval = self.trainer_params['logging']['model_save_interval']
                img_save_interval = self.trainer_params['logging']['img_save_interval']

                if epoch > 1:  # save latest images, every epoch
                    self.logger.info("Saving log_image")
                    self.save_last_image()

                if all_done or (epoch % model_save_interval) == 0:
                    # validation
                    score_optimal, score_student, gap = self.greedy_test_one_batch(greedy_test_episodes,
                                                                                   greedy_test_batch)
                    best_gap = np.array(save_gap)[:, 3].min()

                    if gap < best_gap:
                        self.save_best_model(epoch)
                        best_epoch_result_folder[0] = epoch
                        best_epoch_result_folder[1] = self.result_folder

                    # print('current model epoch and gap', gap)

                    current_epoch_result_folder[0] = epoch
                    current_epoch_result_folder[1] = self.result_folder

                    save_gap.append([epoch, score_optimal, score_student, gap])
                    np.savetxt(self.result_folder + '/gap.txt', save_gap, delimiter=',', fmt='%s')

                if all_done or (epoch % img_save_interval) == 0:
                    self.save_last_image(epoch=epoch)

                if all_done:
                    self.logger.info(" *** Training Done *** ")
                    util_print_log_array(self.logger, self.result_log)
            torch.save(self.env.datas, self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][1])
            torch.save(self.best_student_list, self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][2])

            self.first_time_repair = False

    def suffle_data(self):

        for i in range(self.type_number):
            problem_size_type_ = self.problem_sizes[i]

            episode_type_ = self.episodes[i]

            index = torch.randperm(episode_type_).long()

            self.env.datas[str(problem_size_type_) + '_' + str(episode_type_)] = \
                self.env.datas[str(problem_size_type_) + '_' + str(episode_type_)][index]

            self.best_student_list[str(problem_size_type_) + '_' + str(episode_type_)] = \
                self.best_student_list[
                    str(problem_size_type_) + '_' + str(episode_type_)][index]
        return

    def generate_best_student_list(self):
        self.best_student_list = {}
        if self.first_time_repair:
            for i in range(self.type_number):
                problem_size = self.problem_sizes[i]
                episode_ = self.episodes[i]
                self.best_student_list[str(problem_size) + '_' + str(episode_)] = \
                    torch.arange(problem_size, dtype=torch.int64)[None, :].repeat(episode_, 1)
        else:
            self.best_student_list = torch.load(self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][2],map_location='cuda')
        return

    def repair_all_data(self,best_epoch_result_folder):
        for type in range(self.type_number):
            self.problem_size_type = self.problem_sizes[type]
            self.repair_batch_type = self.trainer_params['repair_batch_size'][type]
            self.episode_type = self.episodes[type]


            # loop2：
            self.repair_one_epoch(best_epoch_result_folder)
        return

    def save_last_image(self,epoch=None):
        if epoch is not None:

            image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=['train_score'])
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                           self.result_log, labels=['train_loss'])
        else:
            image_prefix = '{}/latest'.format(self.result_folder)
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=['train_score'])
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                           self.result_log, labels=['train_loss'])

    def save_best_model(self,epoch):
        self.logger.info("Saving trained_model")
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }
        torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))


    def train_one_epoch(self, epoch):
        self.model.mode = 'train'
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = np.sum(self.episodes)
        episode = 0
        loop_cnt = 0

        self.already_visited_episode = torch.zeros(self.type_number)

        while episode < train_num_episode:

            max_problem_size = np.max(self.problem_sizes)
            self.length_of_subpath = max_problem_size

            self.train_batchs = self.trainer_params['train_batch_size']
            for kk in range(self.type_number):
                if kk == 0:
                    if 0 < self.length_of_subpath <= self.problem_sizes[kk]:
                        self.train_batch_type = self.train_batchs[kk]
                        self.current_trained_problem_types = np.array(self.problem_sizes)
                        break
                else:
                    if self.problem_sizes[kk - 1] < self.length_of_subpath <= self.problem_sizes[kk]:
                        self.train_batch_type = self.train_batchs[kk]
                        self.current_trained_problem_types = np.array(self.problem_sizes)[kk:]
                        break

            data_set_num = len(self.current_trained_problem_types)

            average_batch_size = int(self.train_batch_type / data_set_num)

            remain_ = self.train_batch_type - int(average_batch_size*data_set_num)

            self.current_train_batch_s = torch.ones(data_set_num)*average_batch_size

            self.current_train_batch_s[-1] += remain_

            begin_index = self.type_number - data_set_num

            self.already_visited_episode_2 = self.already_visited_episode.clone()

            self.already_visited_episode_2[begin_index:] +=  self.current_train_batch_s

            index1 = self.already_visited_episode_2 > torch.tensor(self.episodes)

            if index1.any():
                self.roll_data( index1)

            self.current_already_visited_episode = self.already_visited_episode[begin_index:]

            remaining = train_num_episode - episode
            batch_size = min(self.train_batch_type, remaining)

            avg_score, score_student_mean, avg_loss = self.train_one_batch()

            score_AM.update(avg_score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            loop_cnt += 1
            self.logger.info(
                'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}'
                .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                        score_AM.avg, score_student_AM.avg, loss_AM.avg))

            self.already_visited_episode[begin_index:] += self.current_train_batch_s

        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, score_student_AM.avg, loss_AM.avg))

        return score_AM.avg, score_student_AM.avg, loss_AM.avg

    def roll_data(self,index1):

        temp_roll_problem_size = torch.tensor(self.problem_sizes)[index1]
        temp_roll_episode = torch.tensor(self.episodes)[index1]

        for iiii in range(len(temp_roll_episode)):
            shift_num = int((temp_roll_episode - self.already_visited_episode[index1][iiii])[iiii].item())

            self.env.datas[str(temp_roll_problem_size[iiii].item()) + '_' + str(temp_roll_episode[iiii].item())] = \
                torch.roll(self.env.datas[
                        str(temp_roll_problem_size[iiii].item()) + '_' + str(temp_roll_episode[iiii].item())],
                    shifts=shift_num,
                    dims=0)
            self.best_student_list[str(temp_roll_problem_size[iiii].item()) + '_' + str(temp_roll_episode[iiii].item())] \
                = torch.roll(self.best_student_list[
                               str(temp_roll_problem_size[iiii].item()) + '_' + str(temp_roll_episode[iiii].item())],
                           shifts=shift_num,
                           dims=0)

        self.already_visited_episode[index1] = 0
        return

    def train_one_batch(self):

        self.model.train()


        Sub_Problems = None
        Sub_Solutions = None

        for iii in range(len(self.current_train_batch_s)):

            problem_size_temp = self.current_trained_problem_types[iii]
            index = np.where(self.problem_sizes == problem_size_temp)[0][0]

            batch_size_temp = int(self.current_train_batch_s[iii])
            episode_temp = int(self.current_already_visited_episode[iii])

            current_best_solution = self.best_student_list[str(problem_size_temp) + '_' + str(self.episodes[index])][
                                    episode_temp:episode_temp + batch_size_temp]

            length_of_subpath = torch.randint(low=4,
                                high=min(problem_size_temp, self.env_params['max_subtour_length']) + 1, size=[1])[0]

            self.env.load_problems(episode_temp, batch_size_temp, problem_size_type = problem_size_temp,
                                   dataset_size=self.episodes[index],
                                   current_best_solution=current_best_solution,fix_length=length_of_subpath,
                                   sub_path=True)
            if Sub_Problems is None:
                Sub_Problems = self.env.problems
                Sub_Solutions = self.env.solution
            else:
                Sub_Problems = torch.cat((Sub_Problems,self.env.problems),dim=0)
                Sub_Solutions = torch.cat((Sub_Solutions,self.env.solution),dim=0)

        self.env.problems = Sub_Problems
        self.env.solution = Sub_Solutions
        self.env.problem_size = self.length_of_subpath
        self.env.batch_size = int(self.current_train_batch_s.sum().item())

        reset_state, _, _ = self.env.reset()

        prob_list = torch.ones(size=(self.env.batch_size, 0))

        state, reward, reward_student, done = self.env.pre_step()

        current_step = 0

        while not done:

            if current_step == 0:
                selected_teacher = self.env.solution[:, -1]
                selected_student = self.env.solution[:, -1]
                prob = torch.ones(self.env.solution.shape[0], 1)
            elif current_step == 1:
                selected_teacher = self.env.solution[:, 0]
                selected_student = self.env.solution[:, 0]
                prob = torch.ones(self.env.solution.shape[0], 1)
            else:
                selected_teacher, prob, probs, selected_student = self.model(state, self.env.selected_node_list,
                                                                             self.env.solution,
                                                                             current_step)
                loss_mean = - prob.log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, norm_type=2)
                self.optimizer.step()

            current_step += 1
            state, reward, reward_student, done = self.env.step(selected_teacher,
                                                                selected_student)
            prob_list = torch.cat((prob_list, prob), dim=1)

        loss_mean = -prob_list.log().mean()
        return 0, 0, loss_mean.item()

    def repair_one_epoch(self, best_epoch_result_folder):

        checkpoint_fullname = '{}/checkpoint-{}.pt'.format(best_epoch_result_folder[1], best_epoch_result_folder[0])
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.mode = 'test'

        self.time_estimator.reset()
        self.time_estimator_2.reset()

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        episode = 0

        test_num_episode = self.episode_type

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.repair_batch_type, remaining)

            score, score_student_mean = self.repair_one_batch(
                episode, batch_size, self.problem_size_type, clock=self.time_estimator_2)

            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)


            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f},".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score, score_student_mean))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test_All Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg - score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100

            torch.save(self.best_student_list, self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][2])

        return

    def repair_one_batch(self, episode, batch_size, problem_size_type, clock=None):

        self.model.eval()

        self.env.load_problems(episode, batch_size,
                               problem_size_type=problem_size_type, dataset_size=self.episode_type, sub_path=False)

        self.origin_problem = self.env.problems


        with torch.no_grad():
            reset_state, _, _ = self.env.reset()

            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            if self.first_time_repair:
                if self.RI_initial:
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

                    while not done:

                        if current_step == 0:
                            selected_teacher = torch.zeros(B_V, dtype=torch.int64)
                            selected_student = selected_teacher

                        else:
                            selected_teacher, _, _, selected_student = self.model(
                                state, self.env.selected_node_list, self.env.solution, current_step)

                        current_step += 1

                        state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)


                    best_select_node_list = self.env.selected_node_list
            else:
                best_select_node_list = self.best_student_list[str(problem_size_type) + '_' + str(self.episode_type)][
                                        episode: episode + batch_size, :]
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            escape_time, _ = clock.get_est_string(1, 1)

            self.logger.info("curr00,  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                escape_time,
                current_best_length.mean().item(), self.optimal_length.mean().item()))

            if self.RI_initial and self.first_time_repair:
                budget = 0
            else:
                budget = self.env_params['budget']

            origin_problem_size = self.origin_problem.shape[1]

            origin_batch_size = batch_size

            length_all = torch.randint(low=4, high= min(origin_problem_size,
                                self.env_params['repair_max_sub_length']) + 1, size=[budget])
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])

            for bbbb in range(budget):
                torch.cuda.empty_cache()

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
                        self.env.destroy_solution(self.env.problems, best_select_node_list, length_all[bbbb])


                before_reward = partial_solution_length

                before_repair_sub_solution = self.env.solution

                self.env.batch_size = before_repair_sub_solution.shape[0]

                current_step = 0

                reset_state, _, _ = self.env.reset()

                state, reward, reward_student, done = self.env.pre_step()


                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, -1]
                        selected_student = self.env.solution[:, -1]
                    elif current_step == 1:
                        selected_teacher = self.env.solution[:, 0]
                        selected_student = self.env.solution[:, 0]
                    else:
                        selected_teacher, prob, _, selected_student = self.model(
                            state, self.env.selected_node_list, self.env.solution, current_step,repair=True)

                    current_step += 1
                    state, reward, reward_student, done = self.env.step(selected_teacher,
                                                                        selected_student)

                ahter_repair_sub_solution = torch.roll(self.env.selected_node_list, shifts=-1, dims=1)

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

                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                escape_time, _ = clock.get_est_string(1, 1)
                gap_temp =  ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
                self.logger.info("step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                    bbbb,gap_temp, escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

        self.best_student_list[str(problem_size_type) + '_' + str(self.episode_type)][episode: episode + batch_size,:] = best_select_node_list

        return self.optimal_length.mean().item(), current_best_length.mean().item()



    def greedy_test_one_batch(self, test_num_episode, origin_batch_size):

        self.model.eval()
        self.model.mode = 'test'

        episode = 0
        opt_length = 0
        student_length = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode

            batch_size = min(origin_batch_size, remaining)

            self.greedy_test_env.load_problems(episode, batch_size,only_test=True,sub_path=False)

            reset_state, _, _ = self.greedy_test_env.reset()

            B_V = batch_size * 1

            current_step = 0
            state, reward, reward_student, done = self.greedy_test_env.pre_step()

            while not done:

                if current_step == 0:
                    selected_teacher = torch.zeros(B_V, dtype=torch.int64)
                    selected_student = selected_teacher

                else:
                    selected_teacher, _, _, selected_student = self.model(state,
                                                                          self.greedy_test_env.selected_node_list,
                                                                          self.greedy_test_env.solution, current_step)

                current_step += 1

                state, reward, reward_student, done = self.greedy_test_env.step(selected_teacher, selected_student)

            student_length += reward_student.mean().item() * batch_size
            opt_length += reward.mean().item() * batch_size
            episode += batch_size
        mean_student_length = student_length / test_num_episode
        mean_opt_length = opt_length / test_num_episode
        gap = (mean_student_length - mean_opt_length) / mean_opt_length * 100

        return mean_opt_length, mean_student_length, gap

