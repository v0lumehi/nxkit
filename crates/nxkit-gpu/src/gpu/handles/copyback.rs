//! Copy-back of textures that the GPU wrote and whose storage the CPU can access.

use super::super::*;

impl Gpu {
    /// Encodes, at the end of `cb` (the last command buffer of the frame), the copy-back of the written textures
    /// whose storage the CPU can access. The result is the textures copied back and their storage ranges.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::gpu) fn copy_back_written(
        &mut self,
        cb: &mut CommandBuffer,
        scratch: &Scratch,
        view: &dyn EncodeView,
        textures: &FastMap<TexId, Res>,
        dirty: &FastMap<TexId, (Dirty, usize)>,
        copied_back: &[TexId],
        (pools, stats): (&GuestPools, &mut EncodeStats),
        prof: Option<&crate::passprof::FrameProf>,
    ) -> (Vec<TexId>, Vec<(u64, u64)>) {
        // The last writer first: of two textures over the same storage, the one written last is copied back.
        let mut order: Vec<(usize, TexId)> = dirty
            .iter()
            .filter(|(_, (d, _))| !d.is_empty())
            .map(|(id, (_, last))| (*last, *id))
            .collect();
        order.sort_unstable_by(|a, b| b.cmp(a));
        let mut items = Vec::new();
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut backings: FastSet<TexId> = FastSet::default();
        let mut in_place: Vec<(TexId, (u64, u64))> = Vec::new();
        let (mut planned, mut over_budget) = (0u64, 0u32);
        for (_, id) in order {
            let (Some(res), Ok(rec)) = (textures.get(&id), view.record(id)) else {
                continue;
            };
            if res.desc.storage == nxkit_metal::Storage::Memoryless || copied_back.contains(&id) {
                continue;
            }
            let Some(flags) = self.copyback.candidate(id, rec) else {
                continue;
            };
            let (start, end) = (rec.address.get(), rec.end());
            if backings.contains(&res.barrier_id)
                || ranges.iter().any(|&(s, e)| s < end && start < e)
            {
                continue;
            }
            if planned + rec.layout.size > crate::copyback::budget() {
                over_budget += 1;
                nxkit_nvn::diag::note(
                    nxkit_nvn::diag::Reason::CopybackOverBudget,
                    format_args!(
                        "{id:?} {:?} {}x{} {} bytes",
                        res.desc.format, res.desc.width, res.desc.height, rec.layout.size
                    ),
                );
                continue;
            }
            backings.insert(res.barrier_id);
            ranges.push((start, end));
            if res.in_place.is_some() {
                // The GPU wrote the guest bytes themselves: nothing to copy, and the CPU may read them once the frame
                // is done, as after a copy-back.
                in_place.push((id, (start, end)));
                continue;
            }
            planned += rec.layout.size;
            items.push((id, rec, res, &dirty[&id].0, flags));
        }
        let args: Vec<_> = items
            .iter()
            .map(|(_, rec, res, d, _)| (*rec, *res, *d))
            .collect();
        let stamp = |cp: &mut nxkit_metal::ComputePass<'_>| {
            if let Some(p) = prof {
                p.end_pass_compute(cp, p.extra_index());
            }
        };
        if let Some(p) = prof {
            p.set_extra(format!(
                "copyback-end {} textures {planned} bytes",
                items.len()
            ));
        }
        let results = self
            .transfer
            .copy_back_many_with(cb, scratch, pools, &args, Some(&stamp));
        let (mut done, mut done_ranges) = (Vec::new(), Vec::new());
        let (mut bytes, mut refused) = (0u64, 0u32);
        for ((id, rec, res, d, flags), r) in items.iter().zip(results) {
            match r {
                Ok(n) => {
                    stats.copybacks += 1;
                    stats.copyback_bytes += n;
                    bytes += n;
                    done.push(*id);
                    done_ranges.push((rec.address.get(), rec.end()));
                    self.barriers
                        .note_read(res.barrier_id, Stages::DISPATCH | Stages::BLIT);
                    self.copyback.log_first(*id, rec, res, *flags, d, n);
                }
                Err(why) => {
                    refused += 1;
                    stats.skip(why);
                    let what = format_args!(
                        "{id:?} {:?} {}x{} layers {} levels {} scaled {} widened {} linear {} stride {}: {why:?}",
                        res.desc.format,
                        res.desc.width,
                        res.desc.height,
                        res.desc.array_layers,
                        res.desc.mip_levels,
                        res.scaled,
                        res.widened,
                        rec.layout.linear,
                        rec.desc.stride
                    );
                    nxkit_nvn::diag::note(nxkit_nvn::diag::Reason::CopybackRefused, what);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if crate::diag("copyback") && crate::diag::first(&N, 100) {
                        eprintln!("DIAG copyback refused {what}");
                    }
                }
            }
        }
        if !items.is_empty() {
            stats.compute_passes += 1;
        }
        for (id, range) in in_place {
            done.push(id);
            done_ranges.push(range);
        }
        self.copyback
            .log_frame(done.len() as u32, bytes, refused, over_budget);
        (done, done_ranges)
    }
}
