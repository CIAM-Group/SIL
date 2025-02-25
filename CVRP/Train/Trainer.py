import torch
from logging import getLogger
from CVRP.Train.VRPEnv import VRPEnv as Env
from CVRP.Train.VRPModel import VRPModel as Model
from utils.utils import *
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch.nn as nn
import random
class VRP_Self_Improver():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 trainer_params,
                 optimizer_params
                 ):

        # save arguments
        self.env_params = env_params
        self.greedy_test_env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

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
        seed = 123
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = device

        self.greedy_test_env_params['data_path'] = self.tester_params['data_path']
        self.greedy_test_env = Env(**self.greedy_test_env_params)

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1

        self.RI_initial = self.env_params['RI_initial']

        if self.RI_initial:
            self.save_best_model(1)
        else:
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):


        self.time_estimator.reset()
        self.time_estimator_2.reset()

        self.first_time_repair = self.trainer_params['first_time_repair']
        self.repair_before_train = self.trainer_params['repair_before_train']
        greedy_test_episodes = self.tester_params['episodes']
        greedy_test_batch = self.tester_params['batch_size']

        improve_iterations = self.trainer_params['improve_iterations']
        epoch_itervel = self.trainer_params['epoch_itervel']

        self.episodes = np.array(self.trainer_params['episodes'])
        self.problem_sizes = np.array(self.trainer_params['problem_sizes'])
        self.type_number = len(self.episodes)

        # load the problems
        self.env.load_raw_data(self.episodes, problem_sizes=self.problem_sizes,
                               first_time_repair=self.first_time_repair,
                               repair=True, device=self.device)

        self.greedy_test_env.load_raw_data(greedy_test_episodes)


        self.generate_best_student_list()

        save_gap = []

        score_optimal, score_student, gap = self.greedy_test_one_batch(greedy_test_episodes, greedy_test_batch)

        save_gap.append([self.start_epoch - 1, score_optimal, score_student, gap])

        if self.RI_initial:
            best_epoch_result_folder = [1, self.result_folder]
            current_epoch_result_folder = [1, self.result_folder]

        else:
            best_epoch_result_folder = [self.tester_params['model_load']['epoch'],
                                        self.tester_params['model_load']['path']]

            current_epoch_result_folder = [self.tester_params['model_load']['epoch'],
                                           self.tester_params['model_load']['path']]

        # loop1 (main loop): self-improved learning iterations
        for II in range(improve_iterations):

            self.II = II

            # Generate the labeled data or improve its quality.

            if (not self.repair_before_train) and (not self.first_time_repair):
                if II == 0:
                    self.logger.info('The labels already exist and do not need to be reconstructed before training')
                    pass
                else:
                    self.logger.info('The labels already exist and is needed to be reconstructed before training')

                    self.repair_all_data(best_epoch_result_folder)
            else:
                self.logger.info('First time repair, or need to refine the labeled data before train.')

                self.repair_all_data(best_epoch_result_folder)

            end_epoch = self.trainer_params['epochs']

            self.start_epoch = best_epoch_result_folder[0] + 1

            # loop2： Train multiple epochs on the data set, saving the best model

            for epoch in range(self.start_epoch, self.start_epoch + epoch_itervel):

                self.shuffle_data()

                train_score, train_student_score, train_loss = self.train_one_epoch(epoch)

                self.result_log.append('train_score', epoch, train_score)
                self.result_log.append('train_student_score', epoch, train_student_score)
                self.result_log.append('train_loss', epoch, train_loss)

                self.scheduler.step()

                ############################
                # Logs & Checkpoint
                ############################
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
                    score_optimal, score_student, gap = self.greedy_test_one_batch(greedy_test_episodes, greedy_test_batch)
                    best_gap = np.array(save_gap)[:, 3].min()

                    if gap < best_gap:
                        best_epoch_result_folder[0] = epoch
                        best_epoch_result_folder[1] = self.result_folder
                        self.save_best_model(epoch)

                    current_epoch_result_folder[0] = epoch
                    current_epoch_result_folder[1] = self.result_folder
                    self.logger.info(f'current model epoch and gap:{gap}')
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

        return

    def generate_best_student_list(self):
        self.best_student_list = {}
        if self.first_time_repair:
            for i in range(self.type_number):
                problem_size = self.problem_sizes[i]
                episode_ = self.episodes[i]
                node_flag = np.array(list(range(1, problem_size + 1)) + list([1]) * problem_size)[None, :].reshape(1,
                                                                                                                   problem_size,
                                                                                                                   2)
                node_flag = torch.tensor(np.tile(node_flag, (episode_, 1, 1)), dtype=torch.int64)
                self.best_student_list[str(problem_size) + '_' + str(episode_)] = node_flag
        else:
            self.best_student_list = torch.load(self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][2],
                                                map_location=self.device)
        return

    def save_best_model(self, epoch):
        self.logger.info("Saving trained_model")
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }
        torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

    def save_last_image(self, epoch=None):
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

    def shuffle_data(self):
        for i in range(self.type_number):
            problem_size_type_ = self.problem_sizes[i]

            episode_type_ = self.episodes[i]

            index = torch.randperm(episode_type_).long()

            data = self.env.datas[str(problem_size_type_) + '_' + str(episode_type_)]

            for key in data.keys():
                data[key] = data[key][index]

            self.env.datas[str(problem_size_type_) + '_' + str(episode_type_)] = data

            self.best_student_list[str(problem_size_type_) + '_' + str(episode_type_)] = \
                self.best_student_list[str(problem_size_type_) + '_' + str(episode_type_)][index]

        return

    def train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.episodes[0]

        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode

            batch_size = min(self.trainer_params['train_batch_size'][0], remaining)

            avg_score, score_student_mean, avg_loss = self.train_one_batch(episode, batch_size, epoch)

            score_AM.update(avg_score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            loop_cnt += 1
            self.logger.info(
                'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}'
                .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode, score_AM.avg,
                        score_student_AM.avg, loss_AM.avg))

        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode, score_AM.avg, score_student_AM.avg,
                                 loss_AM.avg))

        return score_AM.avg, score_student_AM.avg, loss_AM.avg

    def train_one_batch(self, episode, batch_size, epoch):

        self.model.train()

        self.model.mode = 'train'

        best_select_node_list = self.best_student_list[str(self.problem_sizes[0]) + '_' + str(self.episodes[0])][
                                episode:episode + batch_size]

        fix_length = torch.randint(low=4, high=min(self.problem_sizes[0], self.env_params['max_subtour_length']) + 1, size=[1])[0]  # in [4,V]

        self.env.load_problems(episode, batch_size, problem_size_type=self.problem_sizes[0],
                               dataset_size=self.episodes[0], mode='repair',
                               current_best_solution=best_select_node_list, fix_length=fix_length, sub_path=True)

        reset_state, _, _ = self.env.reset()

        loss_list = []

        state, reward, reward_student, done = self.env.pre_step()

        current_step = 0

        while not done:
            if current_step == 0:
                selected_teacher = self.env.solution[:, 0, 0]
                selected_flag_teacher = self.env.solution[:, 0, 1]
                selected_student = selected_teacher
                selected_flag_student = selected_flag_teacher
                loss_mean = torch.tensor(0)
            else:
                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                               raw_data_capacity=self.env.raw_data_capacity)
                loss_mean = loss_node
                self.model.zero_grad()
                loss_mean.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, norm_type=2)
                self.optimizer.step()

            current_step += 1
            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student,
                                                                selected_flag_teacher,
                                                                selected_flag_student)

            loss_list.append(loss_mean)

        loss_mean = torch.tensor(loss_list).mean()

        return 0, 0, loss_mean.item()

    def repair_all_data(self, best_epoch_result_folder):
        for type in range(self.type_number):
            self.problem_size_type = self.problem_sizes[type]
            self.repair_batch_type = self.trainer_params['repair_batch_size'][type]
            self.episode_type = self.episodes[type]
            self.repair_one_epoch(best_epoch_result_folder)
        return

    def repair_one_epoch(self, best_epoch_result_folder):

        checkpoint_fullname = '{}/checkpoint-{}.pt'.format(best_epoch_result_folder[1], best_epoch_result_folder[0])
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.mode = 'test'

        episode = 0
        test_num_episode = self.episode_type

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.repair_batch_type, remaining)

            self.repair_one_batch(
                episode, batch_size, self.problem_size_type, clock=self.time_estimator_2)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}],".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str))

            torch.save(self.best_student_list, self.env_params['data_path_pt'][0] + self.env_params['data_path_pt'][2])

        return

    def repair_one_batch(self, episode, batch_size, problem_size_type, clock=None):

        self.model.eval()

        self.model.mode = 'test'

        self.env.load_problems(episode, batch_size, problem_size_type=problem_size_type, dataset_size=self.episode_type,
                               mode='repair')

        self.origin_problem = self.env.problems

        with torch.no_grad():

            if self.first_time_repair:

                if self.RI_initial:

                    initial_solution = self.env.random_insert(self.origin_problem)
                    best_select_node_list = initial_solution

                else:
                    reset_state, _, _ = self.env.reset()

                    current_solution = self.best_student_list[str(problem_size_type) + '_' + str(self.episode_type)][
                                       episode:episode + batch_size]

                    current_solution = current_solution

                    self.env.solution = current_solution

                    self.optimal_length = self.env._get_travel_distance_2(
                        self.origin_problem, current_solution.reshape(batch_size, problem_size_type, 2))

                    B_V = batch_size * 1

                    current_step = 0

                    state, reward, reward_student, done = self.env.pre_step()

                    while not done:

                        loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity)

                        if current_step == 0:
                            selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                            selected_flag_student = selected_flag_teacher
                        current_step += 1

                        state, reward, reward_student, done = \
                            self.env.step(selected_teacher, selected_student, selected_flag_teacher,
                                          selected_flag_student)

                    print('Get first complete solution!')

                    best_select_node_list = torch.cat((self.env.selected_student_list.reshape(batch_size, -1, 1),
                                                       self.env.selected_student_flag.reshape(batch_size, -1, 1)),
                                                      dim=2)

            else:
                best_select_node_list = self.best_student_list[str(problem_size_type) + '_' + str(self.episode_type)][
                                        episode:episode + batch_size]

                best_select_node_list = best_select_node_list

            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem,
                                                                  best_select_node_list.reshape(batch_size, -1, 2))

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
            length_all = torch.randint(low=4, high=min(origin_problem_size,
                             self.env_params['repair_max_sub_length']) + 1, size=[budget])
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])

            for bbbb in range(budget):

                torch.cuda.empty_cache()

                best_select_node_list = self.env.Rearrange_solution_caller(
                                       self.origin_problem, best_select_node_list)

                self.env.load_problems(episode, batch_size, problem_size_type=problem_size_type,
                                       dataset_size=self.episode_type, mode='repair',
                                       current_best_solution=best_select_node_list)

                best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)

                if self.env_params['PRC']:
                    partial_solution_length, first_node_index, end_node_index, length_of_subpath, \
                    double_solution, origin_sub_solution, index4, factor = \
                        self.env.destroy_solution_PRC(self.env.problems, best_select_node_list,
                                                      length_all[bbbb], first_index_all[bbbb])
                else:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                        self.env.destroy_solution(self.env.problems, best_select_node_list)

                before_repair_sub_solution = self.env.solution

                self.env.batch_size = before_repair_sub_solution.shape[0]

                before_reward = partial_solution_length

                current_step = 0

                reset_state, _, _ = self.env.reset()

                state, reward, reward_student, done = self.env.pre_step()

                while not done:
                    if current_step == 0:

                        selected_teacher = self.env.solution[:, 0, 0]
                        selected_flag_teacher = self.env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher

                    else:
                        _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity)

                    current_step += 1

                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

                ahter_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                       self.env.selected_student_flag.unsqueeze(2)), dim=2)

                after_reward = reward_student

                if self.env_params['PRC']:
                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution_V2(
                        ahter_repair_sub_solution, before_reward, after_reward, double_solution, origin_sub_solution,
                        index4, origin_batch_size, factor)
                else:

                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution(
                        ahter_repair_sub_solution,
                        before_reward, after_reward, first_node_index, length_of_subpath, double_solution)

                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                escape_time, _ = clock.get_est_string(1, 1)

                self.logger.info(
                    " step{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                        bbbb, ((
                                           current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                        escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))

        self.best_student_list[str(problem_size_type) + '_' + str(self.episode_type)][episode:episode + batch_size] = \
            best_select_node_list

        return

    def greedy_test_one_batch(self, test_num_episode, origin_batch_size):

        self.model.eval()
        self.model.mode = 'test'

        episode = 0
        opt_length = 0
        student_length = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode

            batch_size = min(origin_batch_size, remaining)

            self.greedy_test_env.load_problems(episode, batch_size, only_test=True, mode='test')

            reset_state, _, _ = self.greedy_test_env.reset()

            current_step = 0
            state, reward, reward_student, done = self.greedy_test_env.pre_step()

            while not done:

                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.greedy_test_env.selected_node_list, self.greedy_test_env.solution,
                               current_step,
                               raw_data_capacity=self.greedy_test_env.raw_data_capacity)

                current_step += 1

                state, reward, reward_student, done = \
                    self.greedy_test_env.step(selected_teacher, selected_student, selected_flag_teacher,
                                              selected_flag_student)

            student_length += reward_student.mean().item() * batch_size
            opt_length += reward.mean().item() * batch_size
            episode += batch_size

        mean_student_length = student_length / test_num_episode
        mean_opt_length = opt_length / test_num_episode
        gap = (mean_student_length - mean_opt_length) / mean_opt_length * 100

        return mean_opt_length, mean_student_length, gap
