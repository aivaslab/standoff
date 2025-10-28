class End2EndArchitecture(AblationArchitecture):
    def __init__(self, module_configs, random_probs=None, batch_size=256):


        super().__init__(module_configs=module_configs, random_probs=random_probs, batch_size=batch_size)

        self.full = module_configs['full']
        self.shared = module_configs['shared']
        self.split = module_configs['split']
        self.process_opponent_perception = module_configs['opponent_perception']
        self.size_swap = module_configs['size_swap']
        self.full_infer_decisions = module_configs['full_infer_decisions']

        def mk_e2e(output_type):
            return EndToEndModel(arch=self.arch, output_type=output_type, pad=self.pad)

        if self.full:
            self.end2end_model = mk_e2e('multi')
        elif self.shared:
            shared = mk_e2e('multi')
            self.e2e_op_belief = shared
            self.e2e_my_belief = shared
        elif self.split:
            self.e2e_op_belief = mk_e2e('multi')
            self.e2e_my_belief = mk_e2e('multi')
        else:
            self.e2e_op_belief = mk_e2e('multi')
            self.e2e_my_belief = mk_e2e('multi')

        print('built')

    def build_timestep_input(self, treats, vision, presence):
        chunks = []
        T = treats.size(1)
        for t in range(T):
            chunks.append(torch.cat([
                treats[:, t].flatten(start_dim=1),
                vision[:, t].unsqueeze(1),
                presence[:, t].unsqueeze(1)
            ], dim=1))
        return torch.cat(chunks, dim=1)

    def forward(self, perceptual_field: torch.Tensor, additional_input: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        device = perceptual_field.device
        B = perceptual_field.size(0)

        # === symbolic vs raw input path ===
        if self.process_opponent_perception:
            treats_op = self.treat_perception_op(perceptual_field)
            vision_op = self.vision_perception_op(perceptual_field, is_p1=0)
            presence_op = self.presence_perception_op(perceptual_field, is_p1=0)
            treats_my = self.treat_perception_my(perceptual_field)
            vision_my = self.vision_perception_my(perceptual_field, is_p1=1)
            presence_my = self.presence_perception_my(perceptual_field, is_p1=1)
            presence_op_seq = presence_op if presence_op.shape[1] != 1 else presence_op.repeat(1, 5)
            presence_my_seq = presence_my if presence_my.shape[1] != 1 else presence_my.repeat(1, 5)
            inp_op = self.build_timestep_input(treats_op, vision_op, presence_op_seq)
            inp_my = self.build_timestep_input(treats_my, vision_my, presence_my_seq)
        else:
            inp_op = perceptual_field
            inp_my = perceptual_field

        # === FULL unified model ===
        if self.full:
            out_a = self.end2end_model(inp_op)
            if self.size_swap and self.process_opponent_perception:
                treats_swap = treats_op.flip(dims=[2])
                inp_b = self.build_timestep_input(treats_swap, vision_op, presence_op_seq)
                out_b = self.end2end_model(inp_b)
                out = {k: 0.5 * (out_a[k] + out_b[k].flip(dims=[1])) if k in out_a else out_a[k] for k in out_a}
            else:
                out = out_a
            op_belief_t = out.get('op_belief_t', None)
            my_belief_t = out.get('my_belief_t', None)
            op_decision_t = out.get('op_decision_t', None)
            my_decision = out.get('my_decision', None)

            # optionally run through decision path if not full_infer_decisions
            if not self.full_infer_decisions:
                op_belief_vec = op_belief_t[:, :, -1, :]
                op_decision_t = self.op_decision.forward(
                    op_belief_vec, self.null_decision[:B].to(device), self.null_presence[:B].to(device)
                )
                my_belief_vec = my_belief_t[:, :, -1, :]
                my_decision = self.my_decision.forward(
                    my_belief_vec, op_decision_t, self.null_presence[:B].to(device)
                )
            return {
                'op_belief_t': op_belief_t,
                'my_belief_t': my_belief_t,
                'op_decision_t': op_decision_t,
                'my_decision': my_decision
            }

        # === SPLIT / SHARED ===
        op_out = self.e2e_op_belief(inp_op)
        my_out = self.e2e_my_belief(inp_my)
        if self.size_swap and self.process_opponent_perception:
            treats_swap = treats_op.flip(dims=[2])
            inp_op_sw = self.build_timestep_input(treats_swap, vision_op, presence_op_seq)
            out_sw = self.e2e_op_belief(inp_op_sw)
            op_belief_t = 0.5 * (op_out['op_belief_t'] + out_sw['op_belief_t'].flip(dims=[1]))
        else:
            op_belief_t = op_out['op_belief_t']
        my_belief_t = my_out['my_belief_t']

        op_belief_vec = op_belief_t[:, :, -1, :]
        op_decision_t = self.op_decision.forward(
            op_belief_vec, self.null_decision[:B].to(device), self.null_presence[:B].to(device)
        )
        my_belief_vec = my_belief_t[:, :, -1, :]
        my_decision = self.my_decision.forward(
            my_belief_vec, op_decision_t, self.null_presence[:B].to(device)
        )

        return {
            'op_belief_t': op_belief_t,
            'my_belief_t': my_belief_t,
            'op_decision_t': op_decision_t,
            'my_decision': my_decision
        }