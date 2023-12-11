# Pytorch hook

How hooks work in pytorch?

So the main reason hook exist in pytorch is so that you can inject code into parts of the computational flow that you otherwise wouldn't have access to or would be hard to reach. 

And I think this will make more sense once we get into the examples.

So there are two types of hooks you can use, ones you  can add tensors and ones you can add to modules. 

And the first type we're going to look at are the types you can add to tensors. So these are hooks that allow you to access the gradients as they flow through your backwards graph. So let's just go through an axample so we're on the same page with understanding how auto grad works in pytorch.

## Here is the example without hooks, you can see we have a forward graph and a backwards graph.

And, if we want to update the forward graph, or if we want to inspect any of these tensors as they're being computed, we can just write that here we can print out the values of these tensors or we can add in more computations if we want to change the forward computational graph. But once we call e.backward, this whole computation of this gradient getting passed through these nodes is inaccessible to us. An we can't inspect the gradients as they flow through it or change them if we want to. And we only could see what the gradients were. 

## That's where hooks on tensors come in.

they allow us to inspect the gradients as they flow backwards through the graph and potentially change them if we want to.

This is the same example as before. 

**Order dictionary on a tensor..**






























 