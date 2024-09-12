//! Dead-code elimination pass.

use fxhash::FxHashSet;
use waffle::{cfg::CFGInfo, Block, FunctionBody, Operator, Terminator, Value, ValueDef};

fn op_can_be_removed(op: &Operator) -> bool {
    // Pure ops, and also we allow loads and table.gets to be removed
    // too, because we do not need to uphold Wasm trap semantics at
    // this point (we assume the interpreter is a well-behaved
    // non-trapping program). Also allow global.gets to be removed if
    // unused (they technically have a read side-effect but really
    // should be considered pure).
    match op {
        Operator::I32Load { .. }
        | Operator::I32Load8S { .. }
        | Operator::I32Load8U { .. }
        | Operator::I32Load16S { .. }
        | Operator::I32Load16U { .. }
        | Operator::I64Load { .. }
        | Operator::I64Load8S { .. }
        | Operator::I64Load8U { .. }
        | Operator::I64Load16S { .. }
        | Operator::I64Load16U { .. }
        | Operator::I64Load32S { .. }
        | Operator::I64Load32U { .. }
        | Operator::V128Load { .. }
        | Operator::V128Load8x8S { .. }
        | Operator::V128Load8x8U { .. }
        | Operator::V128Load16x4S { .. }
        | Operator::V128Load16x4U { .. }
        | Operator::V128Load32x2S { .. }
        | Operator::V128Load32x2U { .. }
        | Operator::V128Load8Lane { .. }
        | Operator::V128Load8Splat { .. }
        | Operator::V128Load16Lane { .. }
        | Operator::V128Load16Splat { .. }
        | Operator::V128Load32Lane { .. }
        | Operator::V128Load32Splat { .. }
        | Operator::V128Load32Zero { .. }
        | Operator::V128Load64Lane { .. }
        | Operator::V128Load64Splat { .. }
        | Operator::V128Load64Zero { .. } => true,
        Operator::I32DivS
        | Operator::I32DivU
        | Operator::I32RemS
        | Operator::I32RemU
        | Operator::I64DivS
        | Operator::I64DivU
        | Operator::I64RemS
        | Operator::I64RemU => true,
        Operator::I32TruncF32S
        | Operator::I32TruncF32U
        | Operator::I32TruncF64S
        | Operator::I32TruncF64U
        | Operator::I64TruncF32S
        | Operator::I64TruncF32U
        | Operator::I64TruncF64S
        | Operator::I64TruncF64U => true,
        Operator::TableSize { .. } | Operator::MemorySize { .. } => true,
        Operator::GlobalGet { .. } | Operator::TableGet { .. } => true,
        op if op.is_pure() => true,
        _ => false,
    }
}

/// Scan backwards over a block, marking as used the inputs to any
/// instruction that itself is used (or for a branch arg, for which
/// any target's corresponding blockparam is used). Returns `true` if
/// any changes occurred to the used-value set.
fn scan_block(func: &FunctionBody, block: Block, used: &mut FxHashSet<Value>) -> bool {
    let mark_used = |used: &mut FxHashSet<Value>, mut arg: Value| -> bool {
        let mut changed = false;
        changed |= used.insert(arg);
        while let ValueDef::Alias(orig) = &func.values[arg] {
            arg = *orig;
            changed |= used.insert(arg);
        }
        changed
    };

    log::trace!("DCE: scanning {}", block);
    let mut changed = false;

    func.blocks[block].terminator.visit_targets(|target| {
        log::trace!(" -> considering succ {}", target.block);
        let succ_params = &func.blocks[target.block].params;
        for (&arg, &(_, param)) in target.args.iter().zip(succ_params.iter()) {
            if used.contains(&param) {
                log::trace!(
                    "  -> succ blockparam {} is used; marking arg {} used from term on {}",
                    param,
                    arg,
                    block,
                );
                changed |= mark_used(used, arg);
            }
        }
    });
    match &func.blocks[block].terminator {
        Terminator::CondBr { cond: value, .. } | Terminator::Select { value, .. } => {
            log::trace!(" -> marking branch input {} used", value);
            changed |= mark_used(used, *value);
        }
        Terminator::Return { values } => {
            for &value in values {
                log::trace!(" -> marking return value {} used", value);
                changed |= mark_used(used, value);
            }
        }
        Terminator::Br { .. } | Terminator::Unreachable | Terminator::None => {}
    }

    for &inst in func.blocks[block].insts.iter().rev() {
        match &func.values[inst] {
            ValueDef::BlockParam(..) | ValueDef::Alias(..) => {
                // Nothing.
            }
            ValueDef::PickOutput(value, ..) => {
                if used.contains(&inst) {
                    log::trace!(" -> marking pick-output src {} used", value);
                    changed |= mark_used(used, *value);
                }
            }
            ValueDef::Operator(op, args, _) => {
                if !op_can_be_removed(op) {
                    changed |= used.insert(inst);
                }
                if used.contains(&inst) {
                    for &arg in &func.arg_pool[*args] {
                        log::trace!(" -> marking arg {} used from {}", arg, inst);
                        changed |= mark_used(used, arg);
                    }
                }
            }
            ValueDef::Placeholder(..) | ValueDef::None => {
                // Nothing.
            }
        }
    }

    changed
}

pub(crate) fn run(func: &mut FunctionBody, cfg: &CFGInfo) {
    // For any unreachable blocks, empty their contents and
    // terminators, and remove all blockparams (and there will then be
    // no targets with branch args to adjust because only an
    // unreachable block can branch to an unreachable block).
    for (block, block_def) in func.blocks.entries_mut() {
        if cfg.rpo_pos[block].is_none() {
            log::trace!("removing unreachable block {}", block);
            block_def.insts.clear();
            block_def.params.clear();
            block_def.terminator = Terminator::Unreachable;
        }
    }

    // Now compute value uses.
    let mut used = FxHashSet::default();
    for &(_, param) in &func.blocks[func.entry].params {
        used.insert(param);
    }
    loop {
        let mut changed = false;
        for &block in cfg.rpo.values().rev() {
            changed |= scan_block(func, block, &mut used);
        }
        log::trace!("done with all blocks; changed = {}", changed);
        if !changed {
            break;
        }
    }

    // Now delete any values that aren't used from `insts`, `params`
    // and targets' `args`.
    for block in func.blocks.iter() {
        func.blocks[block].insts.retain(|inst| used.contains(inst));
        let mut terminator = std::mem::take(&mut func.blocks[block].terminator);
        terminator.update_targets(|target| {
            for i in (0..target.args.len()).rev() {
                let succ_arg = func.blocks[target.block].params[i].1;
                if !used.contains(&succ_arg) {
                    target.args.remove(i);
                }
            }
        });
        func.blocks[block].terminator = terminator;
    }
    for block_def in func.blocks.values_mut() {
        block_def.params.retain(|(_ty, param)| used.contains(param));
    }

    // Now validate branch arg types against blockparam types.
    for (block, block_def) in func.blocks.entries() {
        block_def.terminator.visit_targets(|target| {
            for (&arg, &(param_ty, param)) in target
                .args
                .iter()
                .zip(func.blocks[target.block].params.iter())
            {
                let arg = func.resolve_alias(arg);
                let arg_ty = func.values[arg].ty(&func.type_pool).unwrap();
                assert_eq!(
                    arg_ty, param_ty,
                    "block arg {} in {} to param {} on {} mismatches type",
                    arg, block, param, target.block
                );
            }
        });
    }
}
