"""
A pytorch memory cache that can be allocated by ConnectionHandler (on cpu) and used over multiple calls to Runtime.

For now, the only purpose of this code is to ensure that allocated memory will be deleted properly.

"""
import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import time
from typing import AsyncContextManager, Dict, Optional, Sequence

import async_timeout
import torch
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from petals.data_structures import Handle
from petals.utils.asyncio import shield_and_wait
from petals.utils.misc import get_size_in_bytes
from petals.server.req_tensor_descr import NewTensorDescriptor
logger = get_logger(__name__)

'''
class Handle:
    def __init__(self, request_id, idx):
        self.request_id = request_id
        self.idx = idx
'''
class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    def __init__(self, max_size_bytes: Optional[int], max_alloc_timeout: Optional[float] = None):
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else (2**64 - 1)
        self.max_alloc_timeout = max_alloc_timeout
        self._lock_metadata = mp.Lock()
        self._current_size = mp.Value(ctypes.c_int64, 0, lock=False)
        self._enqueued_size = mp.Value(ctypes.c_int64, 0, lock=True)
        self._handle_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._allocated_tensors_key: Dict[Handle, torch.Tensor] = {}
        self._allocated_tensors_value: Dict[Handle, torch.Tensor] = {}
        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime
        self._lock_acquire_memory = mp.Lock()
        self._memory_freed_event = mp.Event()
        self.handles_to_alloc = mp.Manager().list()
        self.descr_to_alloc = mp.Manager().list()
        self.cached_requests_table = mp.Manager().dict()
        self.request_cache_to_del = mp.Manager().list()

        #self.attention_alloc_table = {}

    @property
    def current_size_bytes(self) -> int:
        return self._current_size.value

    @current_size_bytes.setter
    def current_size_bytes(self, value: int):
        self._current_size.value = value

    @property
    def enqueued_size_bytes(self) -> int:
        return self._enqueued_size.value

    @enqueued_size_bytes.setter
    def enqueued_size_bytes(self, value: int):
        self._enqueued_size.value = value

    @property
    def bytes_left(self) -> int:
        return self.max_size_bytes - self.current_size_bytes

    @property
    def handle_counter(self) -> int:
        return self._handle_counter.value

    @handle_counter.setter
    def handle_counter(self, value: int):
        self._handle_counter.value = value

    @contextlib.asynccontextmanager
    async def allocate_cache(
        self, *descriptors: NewTensorDescriptor, timeout: float
    ) -> AsyncContextManager[Sequence[Handle]]:
        """
        Create a handle that is associated with buffers on unique device. If cache full, raises AllocationFailed.

        :param descriptors: one or more tensors tensor of this size, dtype, etc
        :param timeout: optional maximum time to wait for cache allocation; None (default) means no time limit

        :note: if descriptors reside on different devices, it is expected that they are approximately balanced across devices;
          if not, it will count maximum tensor allocation across devices for the purposes of size limit

        :note: This function should be called by connection handlers, it can be called concurrently from multiple processes.
        Furthermore, it can be called concurrently with at most one use_cache call in runtime.
        """
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert all(descr.device is not None for descr in descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)
        max_alloc_size = self.get_allocation_size(*descriptors)

        gib = 1024**3
        cur_size, max_size = self.current_size_bytes, self.max_size_bytes
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        logger.info(
            f"rpc_inference.wait_for_alloc(size={max_alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({cur_size / max_size * 100:.1f}%)"
        )

        alloc_task = asyncio.create_task(self._schedule_alloc(max_alloc_size, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={max_alloc_size / gib:.2f} GiB)")
            yield handles
        finally:
            self._free(max_alloc_size, alloc_task)

    @contextlib.asynccontextmanager
    async def allocate_cache_new(
        self, *descriptors: NewTensorDescriptor, timeout: float
    ) -> AsyncContextManager[Sequence[Handle]]:
        """
        a modified version of allocate_cache()
        in the original allocate_cache(), descriptors is a tuple of two TensorDescriptors
        in this method however, descriptors is a tuple of two list of TensorDescriptors
        """
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        key_descriptors = descriptors[0]
        value_descriptors = descriptors[1]

        assert all(descr.device is not None for descr in key_descriptors), "please specify allocated devices"
        assert all(descr.device is not None for descr in value_descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)
        max_alloc_size = self.get_allocation_size(*key_descriptors) + self.get_allocation_size(*value_descriptors)

        # for testing
        max_alloc_size = 1

        gib = 1024**3
        cur_size, max_size = self.current_size_bytes, self.max_size_bytes
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        logger.info(
            f"rpc_inference.wait_for_alloc(size={max_alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({cur_size / max_size * 100:.1f}%)"
        )

        alloc_task = asyncio.create_task(self._schedule_alloc_new(max_alloc_size, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={max_alloc_size / gib:.2f} GiB)")
            yield handles
        finally:
            self._free(max_alloc_size, alloc_task)

    @staticmethod
    def get_allocation_size(*descriptors: NewTensorDescriptor) -> int:
        """Return the memory size (bytes) to be allocated on a device. If there are many devices, return maximum"""
        if len(descriptors) == 0:
            return 0
        alloc_size_by_device = {}
        for descr in descriptors:
            tensor_size = descr.numel() * get_size_in_bytes(descr.dtype)
            alloc_size_by_device[descr.device] = alloc_size_by_device.get(descr.device, 0) + tensor_size
        return max(alloc_size_by_device.values())
    
    @staticmethod
    def get_allocation_size_new(*descriptors: NewTensorDescriptor) -> int:
        """a modified version of get_allocation_size()"""
        alloc_size_by_device = {}
        for descr in descriptors:
            tensor_size = descr.numel() * get_size_in_bytes(descr.dtype)
            alloc_size_by_device[descr.device] = alloc_size_by_device.get(descr.device, 0) + tensor_size
        return max(alloc_size_by_device.values())

    async def _schedule_alloc(
        self, alloc_size: int, *descriptors: NewTensorDescriptor, timeout: Optional[float]
    ) -> Sequence[Handle]:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
        """
        try:
            async with self._wait_for_free_memory(alloc_size, timeout):
                with self._lock_metadata:
                    handles = tuple(int(self.handle_counter) + i for i in range(len(descriptors)))
                    self.current_size_bytes += alloc_size
                    self.handle_counter += len(handles)  # note: this will eventually overflow and it is okay
                    self._pipe_send.send((handles, descriptors))
                    return handles
        except TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} (timeout={timeout})")
        
    async def _schedule_alloc_new(
        self, alloc_size: int, *descriptors: NewTensorDescriptor, timeout: Optional[float]
    ) -> Sequence[Handle]:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
        """

        handles = []
        recovered_descr = []
        idx = 0
        with self._lock_metadata:
            self.current_size_bytes += alloc_size
        while idx < len(descriptors):

            key_descriptors = descriptors[idx]
            value_descriptors = descriptors[idx + 1]

            recovered_descr.append(tuple([key_descriptors, value_descriptors]))
            with self._lock_metadata:
                self.descr_to_alloc.append(tuple([key_descriptors, value_descriptors]))
            
            # inside this we get key and value descrs for a SINGLE block
            try:
                async with self._wait_for_free_memory(alloc_size, timeout):
                    with self._lock_metadata:
                        req_key_handles = tuple(int(self.handle_counter) + i for i in range(len(key_descriptors)))
                        self.handle_counter += len(req_key_handles)
                        req_value_handles = tuple(int(self.handle_counter) + i for i in range(len(value_descriptors)))
                        self.handle_counter += len(req_value_handles)
                        if self.handle_counter >= 999999999:
                            self.handle_counter = 0
                        # we update attention_alloc_table in handler.py
                        '''
                        handle_idx = 0
                        for req_key_handle, req_value_handle in zip(req_key_handles, req_value_handles):
                            info = AttentionAllocInfo(req_key_handle, req_value_handle)
                            request_id = key_descriptors[handle_idx].request_id
                            self.attention_alloc_table.update({request_id:info})
                            handle_idx += 1
                        '''
                        req_handles = tuple([req_key_handles, req_value_handles])

                        self.handles_to_alloc.append(req_handles)

                        #self.handle_counter += (len(key_handles) + len(value_handles))  # note: this will eventually overflow and it is okay
                        #self.handle_counter += 2
                        handles.append(req_handles)
                        for key_descriptor in key_descriptors:
                            self.cached_requests_table.update({key_descriptor.request_id:1})
                        
            except TimeoutError:
                raise AllocationFailed(f"Could not allocate {alloc_size} (timeout={timeout})")
            idx += 2
            if idx >= len(descriptors):
                break
        handles = tuple(handles)
        #self._pipe_send.send((handles, recovered_descr))
        return handles

    @contextlib.asynccontextmanager
    async def _wait_for_free_memory(self, alloc_size: int, timeout: Optional[float]):
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        with self._enqueued_size.get_lock():
            self._enqueued_size.value += alloc_size
        allocated = False
        try:
            context_manager = async_timeout.timeout(timeout) if timeout != 0 else contextlib.AsyncExitStack()
            # contextlib.AsyncExitStack() is used as a null context here
            async with context_manager:
                if timeout == 0 and self.current_size_bytes + self.enqueued_size_bytes > self.max_size_bytes:
                    raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                async with enter_asynchronously(self._lock_acquire_memory):
                    if self.current_size_bytes + alloc_size > self.max_size_bytes:
                        if timeout == 0:
                            raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                        elapsed_time = time.perf_counter() - start_time
                        remaining_timeout = max(0.0, timeout - elapsed_time) if timeout is not None else None
                        await loop.run_in_executor(None, self._wait_until_available, alloc_size, remaining_timeout)

                allocated = True
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size
                yield
        except asyncio.TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} within {timeout} seconds")
        finally:
            if not allocated:
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size

    def _free(self, alloc_size: int, alloc_task: asyncio.Task):
        if alloc_task.exception() is not None:
            return
        handles = alloc_task.result()

        with self._lock_metadata:
            self._pipe_send.send((handles, None))  # signal runtime to free these handles
            self.current_size_bytes -= alloc_size
        self._memory_freed_event.set()

    def _wait_until_available(self, allocated_size: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if allocated_size > self.max_size_bytes:
            raise AllocationFailed(
                f"Could not allocate {allocated_size} bytes, max cache size = {self.max_size_bytes} bytes"
            )
        timeout = timeout if timeout != float("inf") else None
        deadline = None if timeout is None else time.perf_counter() + timeout
        while self.current_size_bytes + allocated_size > self.max_size_bytes:
            remaining_time = None if timeout is None else deadline - time.perf_counter()
            if not self._memory_freed_event.wait(remaining_time):
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
                )
            self._memory_freed_event.clear()

    '''
    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        """
        Return one or more tensors previously allocated with allocate_cache,

        :note: This method is called by ModuleBackend in runtime: a single process with NO process parallelism.
        However, runtime may call use_cache concurrently with one or more connection handlers calling allocate_cache
        """
        assert os.getpid() == self.runtime_pid
        # note: this specific function is not concurrent, so you can safely allocate/offload/defragment data here

        # read creation/deletion requests from connection handlers
        while self._pipe_recv.poll():
            recv_handles, recv_data = self._pipe_recv.recv()
            if recv_data is not None:  # create new tensors
                assert len(recv_handles) == len(recv_data)
                for handle, descr in zip(recv_handles, recv_data):
                    self._allocated_tensors[handle] = descr.make_zeros()
                    assert handle in self._allocated_tensors, f"Sanity check failed: no such handle ({handle})"
            else:  # delete tensors by handle
                for handle in recv_handles:
                    if handle not in self._allocated_tensors:
                        logger.warning(
                            f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                        )
                    self._allocated_tensors.pop(handle, None)
        yield tuple(self._allocated_tensors[handle] for handle in handles)
        '''

    @contextlib.contextmanager
    def reclaim_cache_for_finished_requests(self, handles):
        assert os.getpid() == self.runtime_pid
        for handle in handles:
            if handle in self._allocated_tensors_key:
                del self._allocated_tensors_key[handle]
            elif handle in self._allocated_tensors_value:
                del self._allocated_tensors_value[handle]




    @contextlib.contextmanager
    def alloc_cache_for_new_requests(self):
        assert os.getpid() == self.runtime_pid
        # note: this specific function is not concurrent, so you can safely allocate/offload/defragment data here

        # read creation/deletion requests from connection handlers
        if len(self.handles_to_alloc) > 0:
            #recv_handles, recv_datas = self.info_to_alloc
            recv_handles = tuple(self.handles_to_alloc)
            recv_datas = self.descr_to_alloc
            if recv_datas is not None:  # create new tensors
                assert len(recv_handles) == len(recv_datas)
                for recv_handle, recv_data in zip(recv_handles, recv_datas):
                    key_handles = recv_handle[0]
                    value_handles = recv_handle[1]
                    key_descr = recv_data[0]
                    value_descr = recv_data[1]
                    # acclocate key cache
                    # note that self._allocated_tensors is a Dict
                    for handle, descr in zip(key_handles, key_descr):
                        self._allocated_tensors_key[handle] = descr.make_zeros()
                        assert handle in self._allocated_tensors_key, f"Sanity check failed: no such handle ({handle})"
                    # acclocate value cache
                    for handle, descr in zip(value_handles, value_descr):
                        self._allocated_tensors_value[handle] = descr.make_zeros()
                        assert handle in self._allocated_tensors_value, f"Sanity check failed: no such handle ({handle})"
            with self._lock_metadata:
                del self.handles_to_alloc[:]
                del self.descr_to_alloc[:]
            '''
            else:  # delete tensors by handle
                for recv_handle in recv_handles:
                    key_handles = recv_handle[0]
                    value_handles = recv_handle[1]
                    for handle in key_handles:
                        if handle not in self._allocated_tensors_key:
                            logger.warning(
                                f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                            )
                        self._allocated_tensors_key.pop(handle, None)
                    for handle in value_handles:
                        if handle not in self._allocated_tensors_value:
                            logger.warning(
                                f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                            )
                        self._allocated_tensors_value.pop(handle, None)
            '''

    @contextlib.contextmanager
    def use_cache_new(self, *handles: Handle) -> Sequence[torch.Tensor]:
        '''
        assert os.getpid() == self.runtime_pid
        # note: this specific function is not concurrent, so you can safely allocate/offload/defragment data here

        # read creation/deletion requests from connection handlers
        if len(self.handles_to_alloc) > 0:
            #recv_handles, recv_datas = self.info_to_alloc
            recv_handles = tuple(self.handles_to_alloc)
            recv_datas = self.descr_to_alloc
            if recv_datas is not None:  # create new tensors
                assert len(recv_handles) == len(recv_datas)
                total = 0
                for recv_handle, recv_data in zip(recv_handles, recv_datas):
                    key_handles = recv_handle[0]
                    value_handles = recv_handle[1]
                    key_descr = recv_data[0]
                    value_descr = recv_data[1]
                    # acclocate key cache
                    # note that self._allocated_tensors is a Dict
                    for handle, descr in zip(key_handles, key_descr):
                        self._allocated_tensors_key[handle] = descr.make_zeros()

                        num_elements = self._allocated_tensors_key[handle].numel()
                        bytes_per_element = self._allocated_tensors_key[handle].element_size()
                        total_memory_bytes = num_elements * bytes_per_element
                        total += total_memory_bytes
                        #print('single memory usage: {} MB'.format(total_memory_bytes / (1024 ** 2)))

                        assert handle in self._allocated_tensors_key, f"Sanity check failed: no such handle ({handle})"
                    # acclocate value cache
                    for handle, descr in zip(value_handles, value_descr):
                        self._allocated_tensors_value[handle] = descr.make_zeros()

                        num_elements = self._allocated_tensors_value[handle].numel()
                        bytes_per_element = self._allocated_tensors_value[handle].element_size()
                        total_memory_bytes = num_elements * bytes_per_element
                        total += total_memory_bytes

                        assert handle in self._allocated_tensors_value, f"Sanity check failed: no such handle ({handle})"

                total_memory_MB = total / (1024 ** 2)
                print(f"memory space used: {total_memory_MB:.2f} MB")
            else:  # delete tensors by handle
                for recv_handle in recv_handles:
                    key_handles = recv_handle[0]
                    value_handles = recv_handle[1]
                    for handle in key_handles:
                        if handle not in self._allocated_tensors_key:
                            logger.warning(
                                f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                            )
                        self._allocated_tensors_key.pop(handle, None)
                    for handle in value_handles:
                        if handle not in self._allocated_tensors_value:
                            logger.warning(
                                f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                            )
                        self._allocated_tensors_value.pop(handle, None)
            with self._lock_metadata:
                del self.handles_to_alloc[:]
                del self.descr_to_alloc[:]
            '''
        handles = handles[0]
        key_handles = tuple(handles[0])
        value_handles = tuple(handles[1])
        yield tuple([[self._allocated_tensors_key[handle] for handle in key_handles], [self._allocated_tensors_value[handle] for handle in value_handles]])


class AllocationFailed(Exception):
    pass
